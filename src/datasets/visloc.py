import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _read_sat_bounds(root: Path, flight_id: str) -> tuple[float, float, float, float]:
  """Returns (lat_min, lon_min, lat_max, lon_max) from satellite_coordinates_range.csv."""
  df = pd.read_csv(root / "satellite_coordinates_range.csv")
  row = df[df["mapname"].isin([f"satellite{flight_id}.tif", f"{flight_id}.tif"])].iloc[0]
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

  `chunk_pixels` specifies the size of each chunk to be extracted from the original satellite image.
  `output_size` acts as a resampling factor of some sort, the selected chunk is resized to it.
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

    sat_path = root / flight_id / f"satellite{flight_id}.tif"
    with rasterio.open(sat_path) as src:
      data = src.read([1, 2, 3])  # (3, H, W)
    self._img = np.transpose(data, (1, 2, 0))  # (H, W, 3) uint8
    orig_h, orig_w = self._img.shape[:2]

    lat_min, lon_min, lat_max, lon_max = _read_sat_bounds(root, flight_id)

    self._chunks: list[tuple[int, int, float, float]] = []

    for y in range(0, orig_h - chunk_pixels, stride_pixels):
      for x in range(0, orig_w - chunk_pixels, stride_pixels):
        cx = x + chunk_pixels // 2
        cy = y + chunk_pixels // 2
        lat = lat_max - (cy / orig_h) * (lat_max - lat_min)
        lon = lon_min + (cx / orig_w) * (lon_max - lon_min)
        self._chunks.append((x, y, lat, lon))

  @property
  def chunk_coords(self) -> list[tuple[float, float]]:
    """List of (lat, lon) center coordinates for each chunk."""
    return [(lat, lon) for _, _, lat, lon in self._chunks]

  def __len__(self) -> int:
    return len(self._chunks)

  def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
    x, y, lat, lon = self._chunks[idx]
    crop = self._img[y : y + self.chunk_pixels, x : x + self.chunk_pixels]
    img = Image.fromarray(crop).resize((self.output_size, self.output_size), Image.LANCZOS)
    if self.transform is not None:
      img = self.transform(img)
    return img, lat, lon


_random_90 = transforms.Lambda(lambda img: TF.rotate(img, random.choice([0, 90, 180, 270])))

# A transform that aims to simulate a domain gap shift from satellite to UAV views in the VisLoc dataset. Not exactly CAEVL-style augmentation, though.
train_sat_uav_sim_transforms = transforms.Compose(
  [
    transforms.Resize(512),  # we expect the input image to be 512x512 to apply geometric augmentations.
    transforms.RandomPerspective(distortion_scale=0.2, p=0.1),  # nadir only
    _random_90,
    transforms.RandomRotation(degrees=30),
    transforms.CenterCrop(370),  # 512/(cos30°+sin30°)≈374, so 370 avoids black corners
    transforms.Resize((256, 256)),
    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.9),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)

inference_sat_transforms = transforms.Compose(
  [
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)

inference_uav_transforms = transforms.Compose(
  [
    transforms.Resize(256),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)
