import os
import time
import warnings
from pathlib import Path

import matplotlib
from dotenv import load_dotenv

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm import tqdm

from src.backbone import DiffusionSatBackbone
from src.datasets.visloc import (
  SatChunkDataset,
  UAVDataset,
  inference_sat_transforms,
  inference_uav_transforms,
)
from src.embedders import PoolConcatEmbedder, normalize_embeddings
from src.evaluation import build_ground_truth, calculate_metrics
from src.ldm_extractor import LDMExtractorCfg
from src.retrievers import FAISSRetriever

warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=".*Importing from cross_attention is deprecated.*")

load_dotenv(Path(__file__).parent / ".env")

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
VISLOC_ROOT = Path(os.environ["VISLOC_ROOT"])
DIFFUSIONSAT_256_CHCKPT = Path(os.environ["DIFFUSIONSAT_256_CHCKPT"])

FLIGHT_ID = "03"
SAT_SCALE = 0.25
CHUNK_SIZE = 512
CHUNK_STRIDE = 128
BATCH_SIZE = 256

assert VISLOC_ROOT.exists(), f"VISLOC_ROOT path {VISLOC_ROOT} does not exist"


t0 = time.perf_counter()
print("Loading backbone...")
backbone = DiffusionSatBackbone(DIFFUSIONSAT_256_CHCKPT, DEVICE, DTYPE)
print(f"Backbone loaded. ({time.perf_counter() - t0:.1f}s)")

print("Building datasets...")
gallery_dataset = SatChunkDataset(
  VISLOC_ROOT,
  FLIGHT_ID,
  chunk_pixels=CHUNK_SIZE,
  stride_pixels=CHUNK_STRIDE,
  scale_factor=SAT_SCALE,
  transform=inference_sat_transforms,
)
gallery_loader = torch.utils.data.DataLoader(
  gallery_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

uav_dataset = UAVDataset(VISLOC_ROOT, FLIGHT_ID, transform=inference_uav_transforms)
uav_loader = torch.utils.data.DataLoader(
  uav_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Gallery: {len(gallery_dataset)} satellite chunks")
print(f"Query:   {len(uav_dataset)} UAV images")


# Precompute VAE latents for gallery and query. Done once; reused across all grid configs.
def precompute_latents(loader, desc="Latents"):
  latents_list, lats_list, lons_list = [], [], []
  with torch.inference_mode():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for imgs, lats, lons in tqdm(loader, desc=desc):
      imgs = imgs.to(DEVICE, dtype=DTYPE)
      latents_list.append(backbone.vae.encode(imgs).latent_dist.sample(generator=g) * 0.18215)
      lats_list.append(lats.numpy())
      lons_list.append(lons.numpy())
  return latents_list, np.concatenate(lats_list), np.concatenate(lons_list)


t0 = time.perf_counter()
print("Pre-computing VAE latents...")
gallery_latents, gallery_lats, gallery_lons = precompute_latents(gallery_loader, "Gallery latents")
uav_latents, uav_lats, uav_lons = precompute_latents(uav_loader, "UAV latents")
print(f"Latents ready. ({time.perf_counter() - t0:.1f}s)")

t0 = time.perf_counter()
print("Building ground truth...")
uav_coords = np.stack([uav_lats, uav_lons], axis=1)
ground_truth = build_ground_truth(uav_coords, gallery_dataset.chunk_bboxes)
print(
  f"Ground truth built: {len(ground_truth)} UAV queries, avg {np.mean([len(gt) for gt in ground_truth]):.1f} matching chunks each. ({time.perf_counter() - t0:.1f}s)"
)


# Grid search over timesteps and layer configs.


def extract_embeddings(latents_list, embedder):
  embeddings = []
  with torch.inference_mode():
    for latents_batch in tqdm(latents_list, desc="Extracting features"):
      batch_feats, _ = backbone.ldm_extractor.forward(latents_batch)
      batch_embs = embedder(batch_feats)
      embeddings.append(batch_embs.cpu())
  return normalize_embeddings(torch.cat(embeddings, dim=0))


grid = [
  # (num_timesteps, save_timesteps)
  # max_i = num_timesteps - min(save_timesteps) = UNet forward passes per batch.
  #
  # Round 1 established:
  #   - nt=10 is 5× faster than nt=50 with no quality loss
  #   - [7] at nt=10 matches [42]/[48,46,42] at nt=50 in R@1 (best overall)
  #   - Too clean ([9]) or too noisy ([3],[5]) both hurt quality
  #   - [9,8,6] spread was worse than [7] alone — wide spread hurts at nt=10
  #
  # Round 2: fine-grained around index 7, tight multi-timestep combos, nt=25 midpoint.
  # # More aggressive feature extraction.
  # (3, [2]),
  # (5, [4]),
  # (5, [3]),
  # (5, [2]),
  # # Fine-grained single-step around the [7] sweet spot.
  # (10, [8]),        # max_i=2 — slightly cleaner than [7]
  # (10, [6]),        # max_i=4 — slightly noisier than [7]
  # # Tight multi-timestep combos at nt=10 (all passes are cheap here).
  (10, [8, 7]),  # max_i=3
  # (10, [7, 6]),     # max_i=4
  # (10, [8, 7, 6]),  # max_i=4, tight cluster around best region
  # # nt=25 midpoint — does a denser schedule improve quality over nt=10?
  # # Proportional mapping: index/nt ≈ same noise level. [7]/10 ≈ [17]/25 ≈ [35]/50.
  # (25, [17]),        # max_i=8, proportional to [7] at nt=10
  # (25, [19, 18, 17]),  # max_i=6, proportional to [8,7,6] cluster
  # (25, [21]),        # max_i=4, proportional to [42] at nt=50
]

layer_idxs = {"up_blocks": {"attn1": "all"}}

configs = grid  # list of (num_timesteps, save_timesteps)
print(f"Starting grid search ({len(configs)} configs)...")

results_path = Path("2-evaluate-timesteps.csv")

results = pd.read_csv(results_path).to_dict(orient="records") if results_path.exists() else []

best_r1 = -1.0
best_gallery_embs = None
best_uav_embs = None
best_preds = None

for idx, (num_timesteps, save_timesteps) in enumerate(configs):
  t_step = time.perf_counter()
  print(f"Grid search step {idx + 1}/{len(configs)}: num_timesteps={num_timesteps}, save_timesteps={save_timesteps}")

  print("  Setting extractor...")
  cfg = LDMExtractorCfg(save_timesteps=save_timesteps, num_timesteps=num_timesteps, layer_idxs=layer_idxs)
  backbone.set_ldm_extractor_cfg(cfg)

  embedder = PoolConcatEmbedder(
    feature_dims=backbone.ldm_extractor.collected_dims,
    save_timesteps=save_timesteps,
  )

  t_emb = time.perf_counter()
  print("  Extracting gallery embeddings...")
  gallery_embs = extract_embeddings(gallery_latents, embedder)
  print("  Extracting UAV embeddings...")
  uav_embs = extract_embeddings(uav_latents, embedder)
  t_emb_elapsed = time.perf_counter() - t_emb

  retriever = FAISSRetriever(gallery_embs)
  _, preds = retriever.search(uav_embs, k=10)

  metrics = calculate_metrics(preds, ground_truth)
  t_step_elapsed = time.perf_counter() - t_step
  results.append(
    {
      "save_timesteps": str(save_timesteps),
      "num_timesteps": cfg.num_timesteps,
      "layer_idxs": str(layer_idxs),
      "emb_dim": embedder.embedding_dim,
      "n_gallery": len(gallery_dataset),
      "n_query": len(uav_dataset),
      "avg_gt_chunks": round(float(np.mean([len(gt) for gt in ground_truth])), 2),
      "normalize_embeddings": True,
      "elapsed_s": round(t_step_elapsed, 1),
      **metrics,
    }
  )
  print(
    f"  → R@1={metrics['Recall@1']:.4f}  R@5={metrics['Recall@5']:.4f}  R@10={metrics['Recall@10']:.4f}  (embed {t_emb_elapsed:.1f}s, total {t_step_elapsed:.1f}s)"
  )

  pd.DataFrame(results).sort_values("Recall@1", ascending=False).to_csv("2-evaluate-timesteps.csv", index=False)

  if metrics["Recall@1"] > best_r1:
    best_r1 = metrics["Recall@1"]
    best_gallery_embs = gallery_embs
    best_uav_embs = uav_embs
    best_preds = preds

print("Grid search complete.")
df = pd.DataFrame(results).sort_values("Recall@1", ascending=False)
print(df.to_string())


# --- t-SNE of best config embeddings ---
assert best_gallery_embs is not None and best_uav_embs is not None and best_preds is not None
t0 = time.perf_counter()
print("Generating embeddings t-SNE...")

rng = np.random.default_rng(42)
n_display = 300
g_idx = rng.choice(len(best_gallery_embs), min(n_display, len(best_gallery_embs)), replace=False)
q_idx = rng.choice(len(best_uav_embs), min(n_display, len(best_uav_embs)), replace=False)

all_embs = np.concatenate([best_gallery_embs[g_idx].numpy(), best_uav_embs[q_idx].numpy()])
coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_embs)
g_coords, q_coords = coords[: len(g_idx)], coords[len(g_idx) :]
g_idx_to_pos = {int(orig): pos for pos, orig in enumerate(g_idx)}

fig, ax = plt.subplots(figsize=(14, 10))
for q_pos, q_orig in enumerate(q_idx):
  for gt_chunk_idx in ground_truth[int(q_orig)]:
    if gt_chunk_idx in g_idx_to_pos:
      g_pos = g_idx_to_pos[gt_chunk_idx]
      ax.plot(
        [q_coords[q_pos, 0], g_coords[g_pos, 0]],
        [q_coords[q_pos, 1], g_coords[g_pos, 1]],
        color="gray",
        linewidth=0.5,
        alpha=0.35,
        zorder=1,
      )
ax.scatter(
  *g_coords.T,
  c="#4ECDC4",
  s=60,
  alpha=0.7,
  edgecolors="black",
  linewidth=0.3,
  zorder=2,
  label="Gallery (satellite chunk)",
)
ax.scatter(
  *q_coords.T,
  c="#FF6B6B",
  s=60,
  alpha=0.7,
  edgecolors="black",
  linewidth=0.3,
  zorder=2,
  marker="^",
  label="Query (UAV image)",
)
ax.legend(fontsize=11)
ax.set_title(f"t-SNE of DiffusionSat Embeddings — best config R@1={best_r1:.4f}", fontsize=14)
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2-evaluate-timesteps-best-embeddings.png", dpi=150)
plt.close()
print(f"Saved 2-evaluate-timesteps-best-embeddings.png ({time.perf_counter() - t0:.1f}s)")


# --- Top-k retrieval grid for hardest queries ---
t0 = time.perf_counter()
print("Generating retrieval visualization...")

display_transform = transforms.Compose(
  [
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
  ]
)
print("Loading display datasets...")
display_uav = UAVDataset(VISLOC_ROOT, FLIGHT_ID, transform=display_transform)
display_gallery = SatChunkDataset(
  VISLOC_ROOT,
  FLIGHT_ID,
  chunk_pixels=CHUNK_SIZE,
  stride_pixels=CHUNK_STRIDE,
  scale_factor=SAT_SCALE,
  transform=display_transform,
)

gt_sims = []
for i, gt_list in enumerate(ground_truth):
  q_emb = best_uav_embs[i].numpy()
  g_embs = best_gallery_embs[list(gt_list)].numpy()
  gt_sims.append(float(np.max(q_emb @ g_embs.T)))
gt_sims = np.array(gt_sims)

n_queries, top_k = 6, 5
query_indices = np.argsort(gt_sims)[:n_queries]

fig, axes = plt.subplots(n_queries, top_k + 1, figsize=(2.5 * (top_k + 1), 2.5 * n_queries))
fig.suptitle(f"UAV query → Top-{top_k} satellite retrievals  (green=correct, red=incorrect)", fontsize=12, y=1.01)

for row, q_idx in enumerate(query_indices):
  uav_img, uav_lat, uav_lon = display_uav[q_idx]
  gt_set = set(ground_truth[q_idx])

  ax = axes[row, 0]
  ax.imshow(uav_img.permute(1, 2, 0).numpy())
  ax.set_title(f"UAV {q_idx}\n({uav_lat:.5f}, {uav_lon:.5f})\nsim={gt_sims[q_idx]:.3f}", fontsize=7)
  ax.axis("off")

  for col in range(top_k):
    chunk_idx = int(best_preds[q_idx, col])
    sat_img_t, sat_lat, sat_lon = display_gallery[chunk_idx]
    correct = chunk_idx in gt_set

    ax = axes[row, col + 1]
    ax.imshow(sat_img_t.permute(1, 2, 0).numpy())
    for spine in ax.spines.values():
      spine.set_edgecolor("green" if correct else "red")
      spine.set_linewidth(3)
    ax.set_title(f"#{col + 1} chunk {chunk_idx}\n({sat_lat:.5f}, {sat_lon:.5f})", fontsize=7)
    ax.axis("off")

plt.tight_layout()
plt.savefig("2-evaluate-timesteps-matches.png", dpi=150)
plt.close()
print(f"Saved 2-evaluate-timesteps-matches.png ({time.perf_counter() - t0:.1f}s)")
