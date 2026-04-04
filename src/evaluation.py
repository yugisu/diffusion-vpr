import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from src.utils import flat_earth_dist_m


def build_ground_truth(
  uav_coords: np.ndarray,
  chunk_bboxes: list[tuple[float, float, float, float]],
) -> list[list[int]]:
  """For each UAV image, return indices of chunks whose bbox contains the UAV GPS point,
  sorted by distance from the UAV point to the chunk center.

  Falls back to the single closest chunk if the point falls outside all bboxes.

  Args:
    uav_coords:   (N, 2) array of (lat, lon) for each UAV query image.
    chunk_bboxes: List of (lat_min, lon_min, lat_max, lon_max) per chunk.
  """
  bboxes = np.array(chunk_bboxes)  # (M, 4)
  lat_mins, lon_mins, lat_maxs, lon_maxs = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
  center_lats = (lat_mins + lat_maxs) / 2
  center_lons = (lon_mins + lon_maxs) / 2
  ground_truth = []
  for lat, lon in uav_coords:
    mask = (lat_mins <= lat) & (lat <= lat_maxs) & (lon_mins <= lon) & (lon <= lon_maxs)
    indices = np.where(mask)[0]
    if len(indices) == 0:
      dists = flat_earth_dist_m(lat, lon, center_lats, center_lons)
      indices = np.array([np.argmin(dists)])
    else:
      dists = flat_earth_dist_m(lat, lon, center_lats[indices], center_lons[indices])
      indices = indices[np.argsort(dists)]
    ground_truth.append(indices.tolist())
  return ground_truth


def recall_at_k(preds: np.ndarray, ground_truth: list[list[int]], k: int) -> float:
  """Recall@k with multi-match ground truth."""
  hits = sum(any(p in gt for p in preds[i, :k]) for i, gt in enumerate(ground_truth))
  return hits / len(ground_truth)


def calculate_metrics(preds: np.ndarray, ground_truth: list[list[int]]) -> dict:
  return {
    "Recall@1": recall_at_k(preds, ground_truth, k=1),
    "Recall@5": recall_at_k(preds, ground_truth, k=5),
    "Recall@10": recall_at_k(preds, ground_truth, k=10),
  }


def plot_tsne(gallery: np.ndarray, query: np.ndarray) -> None:
  # fmt: off
  acq_colors = ["#FF6B6B", "#4ECDC4"]

  all_embeddings = np.concatenate([gallery, query])
  coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_embeddings)

  n = len(gallery)
  gallery_coords, query_coords = coords[:n], coords[n:]

  fig, ax = plt.subplots(figsize=(14, 10))

  for g_pt, q_pt in zip(gallery_coords, query_coords):
    ax.plot(*zip(g_pt, q_pt), color="gray", alpha=0.3, linewidth=1, zorder=1)
    ax.scatter(*g_pt, c=acq_colors[0], s=150, alpha=0.8, edgecolors="black", linewidth=0.5, zorder=2)
    ax.scatter(*q_pt, c=acq_colors[1], s=150, alpha=0.8, edgecolors="black", linewidth=0.5, zorder=2)

  legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=acq_colors[0], markersize=10,
           markeredgecolor="black", markeredgewidth=0.5, label="Gallery"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=acq_colors[1], markersize=10,
           markeredgecolor="black", markeredgewidth=0.5, label="Query"),
    Line2D([0], [0], color="gray", linewidth=1, alpha=0.3, label="Matched pair"),
  ]
  ax.legend(handles=legend_elements, loc="upper right", fontsize=11)
  ax.set_title("t-SNE of Embeddings\n(colors = gallery/query, lines = same location)", fontsize=14)
  ax.set_xlabel("t-SNE Dimension 1")
  ax.set_ylabel("t-SNE Dimension 2")
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()
