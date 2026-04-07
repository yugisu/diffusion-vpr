import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torchvision.transforms.functional as TF
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge as rasterio_merge
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
  """VisLoc UAV drone images. Used as the query set for evaluation only - never for training."""

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
  `scale_factor` is the downsampling factor applied to the original satellite image when loading it into RAM. The chunk coordinates are computed based on the original image size (before downsampling) and the satellite bounds, so they always reflect the correct lat/lon regardless of the scale factor.

  Smaller `scale_factor` -> less "zoomed in" image, given `chunk_pixels` remain the same.
  """

  def __init__(
    self,
    root: Path,
    flight_id: str,
    chunk_pixels: int = 512,
    stride_pixels: int | None = 128,
    scale_factor: float = 0.25,
    transform=None,
  ):
    self.chunk_pixels = chunk_pixels
    self.scale_factor = scale_factor
    self.transform = transform

    if stride_pixels is None:
      stride_pixels = chunk_pixels

    sat_path = root / flight_id / f"satellite{flight_id}.tif"
    tile_paths = sorted((root / flight_id).glob(f"satellite{flight_id}_*.tif"))

    if sat_path.exists():
      with rasterio.open(sat_path) as src:
        h = int(src.height * scale_factor)
        w = int(src.width * scale_factor)
        data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)
    elif tile_paths:
      srcs = [rasterio.open(p) for p in tile_paths]
      native_res = srcs[0].res  # (row_res, col_res) in CRS units
      target_res = (native_res[0] / scale_factor, native_res[1] / scale_factor)
      merged, _ = rasterio_merge(srcs, res=target_res, indexes=[1, 2, 3], resampling=Resampling.bilinear)
      for s in srcs:
        s.close()
      data = merged  # (3, H, W) uint8
      h, w = data.shape[1], data.shape[2]
    else:
      raise FileNotFoundError(f"No satellite TIF found for flight {flight_id}")

    self._img = np.transpose(data, (1, 2, 0))  # (H, W, 3) uint8
    self._bounds = _read_sat_bounds(root, flight_id)

    lat_min, lon_min, lat_max, lon_max = self._bounds

    self._chunks: list[tuple[int, int, float, float]] = []
    self._bboxes: list[tuple[float, float, float, float]] = []
    for y in range(0, h - chunk_pixels, stride_pixels):
      for x in range(0, w - chunk_pixels, stride_pixels):
        cx = x + chunk_pixels // 2
        cy = y + chunk_pixels // 2
        lat = lat_max - (cy / h) * (lat_max - lat_min)
        lon = lon_min + (cx / w) * (lon_max - lon_min)
        self._chunks.append((x, y, lat, lon))
        self._bboxes.append(
          (
            lat_max - ((y + chunk_pixels) / h) * (lat_max - lat_min),  # lat_min (bottom edge)
            lon_min + (x / w) * (lon_max - lon_min),  # lon_min (left edge)
            lat_max - (y / h) * (lat_max - lat_min),  # lat_max (top edge)
            lon_min + ((x + chunk_pixels) / w) * (lon_max - lon_min),  # lon_max (right edge)
          )
        )

  @property
  def chunk_coords(self) -> list[tuple[float, float]]:
    """List of (lat, lon) center coordinates for each chunk."""
    return [(lat, lon) for _, _, lat, lon in self._chunks]

  @property
  def chunk_bboxes(self) -> list[tuple[float, float, float, float]]:
    """List of (lat_min, lon_min, lat_max, lon_max) bounding boxes for each chunk."""
    return self._bboxes

  def __len__(self) -> int:
    return len(self._chunks)

  def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
    x, y, lat, lon = self._chunks[idx]
    crop = self._img[y : y + self.chunk_pixels, x : x + self.chunk_pixels]
    img = Image.fromarray(crop)
    if self.transform is not None:
      img = self.transform(img)
    return img, lat, lon


class PairedUAVSatDataset(Dataset):
  """Each UAV image is paired with its GPS-nearest satellite chunk.

  At init, we perform a one-time nearest-neighbour match (L2 on lat/lon)
  between the UAV query set and the satellite chunk database.

  Returns:
      (uav_img, sat_img, lat, lon) where lat/lon are the UAV GPS coordinates.
  """

  def __init__(
    self,
    root: Path,
    flight_id: str,
    uav_transform=None,
    sat_transform=None,
    top_k_positives: int = 3,  # This makes the dataset to serve one of the top-k nearest satellite chunks for each UAV image, selected at random. Provide "1" to choose the closest chunk.
    sat_scale_factor: float = 0.25,
  ):
    self.uav_ds = UAVDataset(root, flight_id, transform=None)
    self.sat_ds = SatChunkDataset(
      root,
      flight_id=flight_id,
      chunk_pixels=512,
      stride_pixels=512 // 4,
      scale_factor=sat_scale_factor,
      transform=None,
    )
    self.uav_transform = uav_transform
    self.sat_transform = sat_transform

    # Match each UAV image to its nearest satellite chunk by GPS (L2 on lat/lon).
    sat_coords = np.array(self.sat_ds.chunk_coords)  # (N, 2)
    uav_coords = np.array(
      [
        (float(self.uav_ds.records.iloc[i]["lat"]), float(self.uav_ds.records.iloc[i]["lon"]))
        for i in range(len(self.uav_ds))
      ]
    )  # (M, 2)
    dists = np.sum((uav_coords[:, None, :] - sat_coords[None, :, :]) ** 2, axis=2)  # (M, N)
    self.sat_top_k = np.argsort(dists, axis=1)[:, :top_k_positives]  # (M, k)

  def __len__(self) -> int:
    return len(self.uav_ds)

  def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
    # UAV image
    uav_row = self.uav_ds.records.iloc[idx]
    uav_img = Image.open(self.uav_ds.drone_dir / uav_row["filename"]).convert("RGB")
    lat, lon = float(uav_row["lat"]), float(uav_row["lon"])

    # Matched satellite chunk — sample uniformly from top-k closest by GPS.
    sat_idx = int(random.choice(self.sat_top_k[idx]))
    sat_x, sat_y, _, _ = self.sat_ds._chunks[sat_idx]
    sat_crop = self.sat_ds._img[sat_y : sat_y + self.sat_ds.chunk_pixels, sat_x : sat_x + self.sat_ds.chunk_pixels]
    sat_img = Image.fromarray(sat_crop)

    if self.uav_transform is not None:
      uav_img = self.uav_transform(uav_img)
    if self.sat_transform is not None:
      sat_img = self.sat_transform(sat_img)

    return uav_img, sat_img, lat, lon


_random_90 = transforms.Lambda(lambda img: TF.rotate(img, random.choice([0, 90, 180, 270])))


inference_uav_transforms = transforms.Compose(
  [
    transforms.Resize(256),
    transforms.CenterCrop((256, 256)),
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


# UAV image augmentation for supervised training
train_sup_uav_transforms = transforms.Compose(
  [
    # Geometric
    transforms.Resize(300),
    transforms.RandomResizedCrop(256, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
    _random_90,
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.4),
    # Photometric - UAV images often have different exposure, haze, and sensor response
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)], p=0.8),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    # Simulate motion blur from drone movement
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(1, 7), sigma=(0.1, 1.0))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)

# Satellite augmentation for supervised training
train_sup_sat_transforms = transforms.Compose(
  [
    # Geometric - satellite tiles can appear at any orientation and scale
    transforms.Resize(300),
    transforms.RandomResizedCrop(256, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
    _random_90,
    # Photometric - sensor/season/time-of-day variation
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)], p=0.8),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)
