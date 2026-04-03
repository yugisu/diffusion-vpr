# SSL4EO-S12 Dataset

**Source**: https://github.com/zhu-xlab/SSL4EO-S12

## Structure

100 geographic patches, each with 4 seasonal acquisitions (~3-month intervals across 2020–2021). Per patch/season:

- **S1**: 2-channel SAR (VV, VH), 264×264px, float32
- **S2A/S2C**: 12–13 multispectral bands at mixed resolutions (264×264 at 10m for RGB bands)
- **RGB**: 8-bit PNG true-color composites from S2 (B4-B3-B2), 264×264px

Directory layout:
```
SSL4EO-S12/
├── rgb/        — 8-bit PNG true-color composites (4 per patch)
├── s1/         — Sentinel-1 SAR (VV.tif, VH.tif per acquisition)
├── s2a/        — Sentinel-2 Archive, 12 bands (B1–B12, no B10)
└── s2c/        — Sentinel-2 Consistent, 13 bands (includes B10)
```

Each patch ID is a 7-digit zero-padded integer. Within each patch, subdirectories are named by acquisition timestamp + MGRS tile.

## Relevance to this project

### Positive pair construction

Each patch has 4 seasonal observations of the same location — natural positives without needing UAV data. Same location, different season/lighting = appearance variation the model should be invariant to. This is a real-temporal alternative to `SatSimDataset`'s synthetic pairs from a single GeoTIFF.

### Recommended usage

Use the RGB PNGs (264×264, 8-bit) — most directly compatible with the DiffusionSat backbone (trained on 256×256 RGB). Center-crop or resize to 256×256.

For Stage 2 VICRegL training: sample two random seasonal observations of the same patch as positives, apply `train_sat_transforms` to each. `FuserEmbedder`, `SpatialLoss`, and `FAISSRetriever` require no changes.

### Caveats

- **Scale**: The example has 100 patches (400 RGB images) — too small for SSL. The full dataset has 250K locations; use at minimum the 1000-patch tarball.
- **No UAV simulation**: SSL4EO-S12 is pure satellite. The UAV domain gap still needs to be addressed via `train_sat_uav_sim_transforms` heavy augmentation (or CAEVL's vignetting pipeline) applied to one of the two seasonal views.
- **Resolution**: 10m/px overhead orthoimagery — verify GSD matches the DiffusionSat checkpoint's expected input domain relative to your VisLoc satellite reference.
