import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def flat_earth_dist_m(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
  """Approximate distances in metres from one point to an array of points."""
  dlat = (lats - lat1) * 111_111
  dlon = (lons - lon1) * 111_111 * np.cos(np.radians(lat1))
  return np.sqrt(dlat**2 + dlon**2)


def visualize_embeddings(gallery_embs, query_embs, ground_truth):
  g_idx = range(len(gallery_embs))
  q_idx = range(len(query_embs))

  all_embs = np.concatenate([gallery_embs[g_idx].numpy(), query_embs[q_idx].numpy()])
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
          linewidth=0.1,
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
  ax.set_title("t-SNE Embeddings Visualization", fontsize=14)
  ax.set_xlabel("t-SNE Dimension 1")
  ax.set_ylabel("t-SNE Dimension 2")
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()
