# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**diffusion-vpr** is a research project for Visual Place Recognition (VPR) using diffusion model features. The goal is geolocation by matching UAV drone images (queries) against satellite imagery (gallery). It also supports satellite-to-satellite matching as a simpler baseline.

Current status:
- Zero-shot satellite-to-satellite achieves ~0.79 R@1 on the SeCo dataset
- Zero-shot UAV-to-satellite achieves ~0.02 R@1 on VisLoc — the active research challenge

## Environment Setup

Requires a sibling directory `../SatDiFuser` with an editable install (the `satdifuser` package comes from there).

```bash
uv sync
```

Environment variables (copy `.env.example`):
- `VISLOC_ROOT` — path to VisLoc dataset root
- `DIFFUSIONSAT_256_CHCKPT` — path to DiffusionSat checkpoint

## Commands

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Run a notebook
uv run jupyter lab
```

Ruff config: 2-space indent, 120-char line length, double quotes, import sorting enabled, `E402` ignored, `F401` unfixable (unused imports must be removed manually). Linting applies to `.ipynb` files too.

## Architecture

All experiments live in Jupyter notebooks (numbered `0-*`, `1-*`, `2-*`). The `src/` package contains reusable building blocks.

### Feature Extraction Pipeline

**`src/backbone.py` — `DiffusionSatBackbone`**

Wraps DiffusionSat (Stable Diffusion 2.1 fine-tuned on satellite imagery). The model is **frozen** — it is used only as a feature extractor, not for generation. It takes 6.4 GB VRAM.

Flow: image → VAE encoder → latent → add noise at timesteps [48, 46, 42] → UNet forward pass → hook intermediate attention features from `down_blocks[*].attn1`.

The `LDMExtractorCfg` in `src/ldm_extractor.py` specifies which layers and timesteps to hook. The diffusion mode is `inversion` (DDIM inversion, not random noise). Input images must be 256×256.

### Embedding Strategies

**`src/embedders.py`**

- **`PoolConcatEmbedder`** — zero-shot baseline. Pools all multi-scale, multi-timestep features and concatenates. Large embedding, no learned parameters.
- **`FuserEmbedder`** — trainable. Uses `GlobalWeightedFuser` from SatDiFuser to learn cross-scale/timestep weights, then projects to `projection_dim=384`. This is the active development target.

Both use `gem_pool()` (generalized mean, p=3.0) and `normalize_embeddings()` (L2).

### Datasets

**`src/datasets.py`**

- **`UAVDataset`** — VisLoc drone images with GPS coords. Query-side only.
- **`SatChunkDataset`** — Fixed 256×256 tiles (128px stride) from a satellite GeoTIFF. Gallery-side. Each chunk has GPS coords for ground truth construction.

### Loss

**`src/losses.py` — `SpatialLoss`**

Combined VICReg (global pooled embeddings) + VICRegL (spatial feature matching via nearest-neighbor in L2 space). The `alpha` parameter blends global vs. local components. Supports location-aware matching using GPS coordinates.

### Retrieval & Evaluation

**`src/retrievers.py`** — `FAISSRetriever` using `IndexFlatIP` (inner product = cosine similarity on L2-normalized embeddings).

**`src/evaluation.py`** — GPS-based ground truth construction, Recall@1/5/10, t-SNE visualization.

### Data Transforms

**`src/data_transforms.py`** — Four pipelines:
- `train_sat_transforms` — light augmentation
- `train_sat_uav_sim_transforms` — heavy augmentation to simulate UAV domain gap
- `inference_sat_transforms` / `inference_uav_transforms` — standard normalization (UAV adds center crop)
