import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from torch.utils.data import Dataset


def _read_sat_bounds(root: Path, flight_id: str) -> tuple[float, float, float, float]:
  """Returns (lat_min, lon_min, lat_max, lon_max) from satellite_coordinates_range.csv."""
  df = pd.read_csv(root / "satellite_coordinates_range.csv")
  row = df[df["mapname"] == f"satellite{flight_id}.tif"].iloc[0]
  return (
    float(row["RB_lat_map"]),
    float(row["LT_lon_map"]),
    float(row["LT_lat_map"]),
    float(row["RB_lon_map"]),
  )


class UAVDataset(Dataset):
  """VisLoc UAV drone images. Used as the query set for evaluation only — never for training."""

  def __init__(self, root: Path, flight_id: str, transform=None):
    self.drone_dir = root / flight_id / "drone"
    self.transform = transform
    df = pd.read_csv(root / flight_id / f"{flight_id}.csv")
    self.records = df[["filename", "lat", "lon"]].reset_index(drop=True)

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
    row = self.records.iloc[idx]
    img = Image.open(self.drone_dir / row["filename"]).convert("RGB")
    if self.transform is not None:
      img = self.transform(img)
    return img, float(row["lat"]), float(row["lon"])


class SatChunkDataset(Dataset):
  """Fixed, evenly-tiled satellite chunks for the retrieval reference database.

  The full tiff is loaded into RAM at init; chunks are pure numpy slices.

  At 0.3 m/px satellite GSD, chunk_pixels=256 covers ~76.8m per side —
  large enough to contain any UAV image footprint (25–50m) with margin.
  stride_pixels=128 (50% overlap) ensures every GPS point falls in ≥4 chunks.

  Exposes `gt_chunks` for use with build_ground_truth().

  Args:
    root:          VisLoc dataset root.
    flight_id:     Flight identifier (e.g. "03").
    chunk_pixels:  Crop size in original tiff pixels (controls ground coverage).
    stride_pixels: Stride in original tiff pixels between chunk origins.
    output_size:   Spatial dims of the returned image after resize.
    transform:     Applied to each PIL image before returning.
  """

  def __init__(
    self,
    root: Path,
    flight_id: str,
    chunk_pixels: int = 256,
    stride_pixels: int = 128,
    output_size: int = 256,
    transform=None,
  ):
    self.chunk_pixels = chunk_pixels
    self.output_size = output_size
    self.transform = transform

    lat_min, lon_min, lat_max, lon_max = _read_sat_bounds(root, flight_id)

    sat_path = root / flight_id / f"satellite{flight_id}.tif"
    with rasterio.open(sat_path) as src:
      data = src.read([1, 2, 3])  # (3, H, W)
    self._img = np.transpose(data, (1, 2, 0))  # (H, W, 3) uint8
    orig_h, orig_w = self._img.shape[:2]

    self._chunks: list[tuple[int, int, float, float]] = []
    self._bboxes: list[tuple[float, float, float, float]] = []

    for y in range(0, orig_h - chunk_pixels, stride_pixels):
      for x in range(0, orig_w - chunk_pixels, stride_pixels):
        cx = x + chunk_pixels // 2
        cy = y + chunk_pixels // 2
        lat = lat_max - (cy / orig_h) * (lat_max - lat_min)
        lon = lon_min + (cx / orig_w) * (lon_max - lon_min)
        self._chunks.append((x, y, lat, lon))

        blat_max = lat_max - (y / orig_h) * (lat_max - lat_min)
        blat_min = lat_max - ((y + chunk_pixels) / orig_h) * (lat_max - lat_min)
        blon_min = lon_min + (x / orig_w) * (lon_max - lon_min)
        blon_max = lon_min + ((x + chunk_pixels) / orig_w) * (lon_max - lon_min)
        self._bboxes.append((blat_min, blon_min, blat_max, blon_max))

  @property
  def gt_chunks(self) -> list[dict]:
    """List of {bbox_gps} dicts compatible with build_ground_truth()."""
    return [{"bbox_gps": bb} for bb in self._bboxes]

  def __len__(self) -> int:
    return len(self._chunks)

  def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
    x, y, lat, lon = self._chunks[idx]
    crop = self._img[y : y + self.chunk_pixels, x : x + self.chunk_pixels]
    img = Image.fromarray(crop).resize((self.output_size, self.output_size), Image.LANCZOS)
    if self.transform is not None:
      img = self.transform(img)
    return img, lat, lon
