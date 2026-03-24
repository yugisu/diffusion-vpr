import numpy as np


# For u2s retrieval, each UAV image may match multiple satellite chunks (overlapping tiles).
# ground_truth is a list of sets of valid chunk indices.
def build_ground_truth(
  uav_lats: np.ndarray,
  uav_lons: np.ndarray,
  chunks: list[dict],
) -> list[set]:
  """For each UAV image, find all satellite chunk indices whose GPS bbox contains it.
  Falls back to the nearest chunk if no bbox contains the point."""
  ground_truth = []
  for lat, lon in zip(uav_lats, uav_lons):
    matching = set()
    for i, chunk in enumerate(chunks):
      lat_min, lon_min, lat_max, lon_max = chunk["bbox_gps"]
      if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
        matching.add(i)
    if not matching:
      dists = [(lat - c["lat"]) ** 2 + (lon - c["lon"]) ** 2 for c in chunks]
      matching = {int(np.argmin(dists))}
    ground_truth.append(matching)
  return ground_truth


def recall_at_k(preds: np.ndarray, ground_truth: list[set], k: int) -> float:
  """Recall@k with multi-match ground truth."""
  hits = sum(any(p in gt for p in preds[i, :k]) for i, gt in enumerate(ground_truth))
  return hits / len(ground_truth)


def calculate_metrics(preds: np.ndarray, ground_truth: list[set]) -> dict:
  return {
    "Recall@1": recall_at_k(preds, ground_truth, k=1),
    "Recall@5": recall_at_k(preds, ground_truth, k=5),
    "Recall@10": recall_at_k(preds, ground_truth, k=10),
  }
