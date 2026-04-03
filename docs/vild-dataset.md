# ViLD Dataset Description

## Overview

**ViLD** is a large-scale **visual geo-localization** dataset pairing UAV (drone) images with satellite imagery. The task is: given a UAV query image, find the matching location in the satellite reference imagery using GPS/UTM coordinates.

**Dataset root:** `/home/slymahel/data/ViLD_dataset`  
**Total size:** ~48 GB, ~1,036,000 images across two flights (Flight 09 and Flight 10)

---

## Directory Structure

```
ViLD_dataset/
├── flight09_satellite/   # 607,856 RGB JPEG images (512×512)
├── flight09_uav/         # 59,438 Grayscale JPEG images (512×512)
├── flight10_satellite/   # 337,804 RGB JPEG images (512×512)
├── flight10_uav/         # 30,910 Grayscale JPEG images (512×512)
├── coordinates/          # 4 .pkl files (Lambert93 coords per image)
├── vild/                 # CSV splits for Flight 10
│   ├── vild_coordinates__train.csv   (302K rows)
│   ├── vild_coordinates__val.csv     (18K rows)
│   └── vild_coordinates__test.csv    (48K rows)
└── vild_09/              # CSV splits for Flight 09
    ├── vild_09_coordinates__train.csv (500K rows)
    ├── vild_09_coordinates__val.csv   (47K rows)
    └── vild_09_coordinates__test.csv  (120K rows)
```

---

## Key File Formats

### Pickle files (`coordinates/`) — primary interface for caevl
Dictionaries mapping image filename stems (no extension) to coordinate arrays:
```python
{ "images_undistorted_0000040524": np.array([utm_east, utm_north, angle, ...]) }
```
- Coordinate order: `[utm_east, utm_north]` (x, y). Only the first 2 values are used.
- 4 files: `flight09_satellite.pkl`, `flight09_uav.pkl`, `flight10_satellite.pkl`, `flight10_uav.pkl`

### CSV annotations (for reference / split generation)
Each CSV has 4 columns:
```
filename,             utm_north,          utm_east,           image_type
0000009962_77_2021.jpg, 5324681.91, 479462.17, reference
images_undistorted_0000040524.jpg, 5333761.27, 494967.08, query
```

- `image_type` is either `"query"` (UAV) or `"reference"` (satellite)
- Coordinates are UTM31N (EPSG:32631)
- Note: column order is `utm_north, utm_east` in CSV but pickle stores `[utm_east, utm_north]`

### Satellite image filenames
`{ID}_{collection_code}_{year}.jpg` → e.g. `0000083753_45_2023.jpg`

### UAV image filenames
`images_undistorted_{ID}.jpg` → e.g. `images_undistorted_0000021702.jpg`

### Important: both modalities are used as grayscale
Raw on disk: satellite = RGB (3 channels), UAV = grayscale (1 channel).  
**caevl converts both to grayscale** (`grayscale: True` in all configs) — the model sees 1-channel input for both.

### Normalization stats (JSON)
Located at `flight10_satellite/vol10_1400_5_50_coarse.json` and `flight10_uav/vol10_undistorted.json`:
```json
{"mean": 0.4385530603794264, "std": 0.1633757279907221}
```
Note: `apply_normalization: False` in both training configs — images are kept in `[0, 1]`.

---

## Split Distribution

| Flight | Split | Query (UAV) | Reference (Satellite) | Total |
|--------|-------|-------------|----------------------|-------|
| 10 | train | 25,602 | 276,420 | 302,022 |
| 10 | val | 1,387 | 16,725 | 18,112 |
| 10 | test | 3,920 | 44,658 | 48,578 |
| 09 | train | 42,331 | 457,613 | 499,944 |
| 09 | val | 4,894 | 42,183 | 47,077 |
| 09 | test | 12,213 | 108,060 | 120,273 |

Splits are **spatially separated** by UTM X-axis (75/10/15%), so there is no geographic leakage between train/val/test.

---

## PyTorch Dataset Implementation

These implementations mirror how caevl (`/home/slymahel/caevl`) actually uses the data.

```python
import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rgb_to_grayscale, resize, center_crop
from torchvision.transforms import InterpolationMode

DATASET_ROOT = "/home/slymahel/data/ViLD_dataset"
IMAGE_SIZE   = (256, 224)   # (height, width) — as used in all caevl configs


def load_coords_pickle(pkl_path):
    """Returns dict: stem -> np.array([utm_east, utm_north, ...])"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def preprocess(img: Image.Image, image_size=IMAGE_SIZE) -> torch.Tensor:
    """
    Resize (aspect-preserving) + center crop to image_size, convert to
    grayscale, return float32 tensor in [0, 1] with shape (1, H, W).
    Both satellite (RGB) and UAV (grayscale) images go through this.
    """
    # Aspect-preserving resize so the smaller side matches target
    h_t, w_t = image_size
    w, h = img.size
    scale = max(h_t / h, w_t / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to exact target size
    left  = (new_w - w_t) // 2
    top   = (new_h - h_t) // 2
    img   = img.crop((left, top, left + w_t, top + h_t))

    # Convert to grayscale tensor (1, H, W) in [0, 1]
    img_t = torch.from_numpy(np.array(img)).float() / 255.0
    if img_t.ndim == 3:                  # RGB → grayscale
        img_t = img_t.permute(2, 0, 1)  # (H,W,C) → (C,H,W)
        img_t = rgb_to_grayscale(img_t)  # (1,H,W)
    else:                                # already grayscale
        img_t = img_t.unsqueeze(0)       # (1,H,W)
    return img_t


def apply_canny(img_t: torch.Tensor) -> torch.Tensor:
    """Apply Canny edge detection. Input/output: (1, H, W) float32 in [0, 1]."""
    arr = (img_t.squeeze(0).numpy() * 255).astype(np.uint8)
    edges = cv2.Canny(arr, 100, 200)
    return torch.from_numpy(edges).float().unsqueeze(0) / 255.0


class ViLDDataset(Dataset):
    """
    Single-image dataset for either UAV (query) or satellite (reference) images.

    Loads images from a folder and looks up coordinates from the corresponding
    pickle file. Both modalities are returned as 1-channel grayscale tensors.

    Args:
        image_dir:  path to flight10_uav/, flight10_satellite/, etc.
        pkl_path:   path to the corresponding coordinates pickle file
        canny:      if True, return Canny edge map instead of raw grayscale
        image_size: (H, W) target size after resize+crop
    """
    def __init__(self, image_dir, pkl_path, canny=True, image_size=IMAGE_SIZE):
        self.image_dir  = Path(image_dir)
        self.coords     = load_coords_pickle(pkl_path)
        self.canny      = canny
        self.image_size = image_size

        exts = {".jpg", ".jpeg", ".JPG", ".png", ".PNG", ".tif", ".tiff"}
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix in exts and p.stem in self.coords
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path  = self.images[idx]
        img   = Image.open(path)
        img_t = preprocess(img, self.image_size)   # (1, H, W) float32 in [0,1]
        if self.canny:
            img_t = apply_canny(img_t)

        xy     = self.coords[path.stem]            # [utm_east, utm_north, ...]
        coords = torch.tensor(xy[:2], dtype=torch.float32)  # (utm_east, utm_north)
        return img_t, coords


# --- Usage: training on UAV images (queries) ---
train_ds = ViLDDataset(
    image_dir = f"{DATASET_ROOT}/flight10_uav",
    pkl_path  = f"{DATASET_ROOT}/coordinates/flight10_uav.pkl",
    canny     = True,
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=8, pin_memory=True)

for imgs, coords in train_loader:
    # imgs:   (B, 1, 256, 224)  float32 — Canny edge map
    # coords: (B, 2)            float32 — [utm_east, utm_north]
    pass


# --- Usage: reference gallery (satellite) for retrieval ---
ref_ds = ViLDDataset(
    image_dir = f"{DATASET_ROOT}/flight10_satellite",
    pkl_path  = f"{DATASET_ROOT}/coordinates/flight10_satellite.pkl",
    canny     = True,
)
ref_loader = DataLoader(ref_ds, batch_size=128, shuffle=False,
                        num_workers=8, pin_memory=True)


# --- Usage: splitting into train/val/test via CSV ---
# The CSVs provide the official spatial split. Filter image lists accordingly:
import pandas as pd

def get_split_stems(flight, split):
    """Return set of filename stems belonging to a given split."""
    csv_dir = "vild" if flight == "flight10" else "vild_09"
    prefix  = "vild" if flight == "flight10" else "vild_09"
    csv     = pd.read_csv(f"{DATASET_ROOT}/{csv_dir}/{prefix}_coordinates__{split}.csv")
    return set(Path(f).stem for f in csv.filename)

class ViLDSplitDataset(ViLDDataset):
    """ViLDDataset restricted to a specific train/val/test split."""
    def __init__(self, flight, modality, split, **kwargs):
        assert modality in ("uav", "satellite")
        super().__init__(
            image_dir = f"{DATASET_ROOT}/{flight}_{modality}",
            pkl_path  = f"{DATASET_ROOT}/coordinates/{flight}_{modality}.pkl",
            **kwargs,
        )
        allowed = get_split_stems(flight, split)
        self.images = [p for p in self.images if p.stem in allowed]


# Example: Flight 10, train split, UAV queries
train_ds = ViLDSplitDataset("flight10", "uav", "train", canny=True)
# Example: Flight 10, test split, satellite references
test_ref_ds = ViLDSplitDataset("flight10", "satellite", "test", canny=True)
```

---

## Key Points

1. **Task**: Cross-modal geo-localization — match a grayscale UAV image to its GPS location in a satellite reference gallery.
2. **Both modalities are grayscale** — satellite images (RGB on disk) are converted to 1 channel. The model sees `(1, 256, 224)` tensors for both.
3. **Canny edges are the primary input** — caevl applies Canny edge detection as preprocessing; raw pixel values are not used directly.
4. **No pixel normalization** — images stay in `[0, 1]`; caevl sets `apply_normalization: False`.
5. **Image size is `(256, 224)`** — not the native 512×512; resize + center crop is applied.
6. **Coordinate order**: pickle files store `[utm_east, utm_north]` (x first, y second).
7. **Coordinate system**: pickle files use Lambert93; CSVs use UTM31N (EPSG:32631). caevl reads directly from the pickles.
8. **Evaluation metric**: Recall@k and Accuracy@n meters (n = 25, 50, 100, 150, 250, 500 m) using cosine similarity + FAISS.
9. **Scale**: ~1M images, 48 GB — use `num_workers=8` and `pin_memory=True`; cache descriptors at eval time.
