# CAEVL

**Paper**: "Beyond Paired Data: Self-Supervised UAV Geo-Localization from Reference Imagery Alone" (WACV 2026)
**arXiv**: https://arxiv.org/abs/2512.02737

## Training process

CAEVL trains using **only satellite reference imagery** — no paired UAV-satellite data required. The domain gap is handled entirely via augmentation simulating UAV visual characteristics.

All images (satellite during training, both satellite and UAV at inference) are preprocessed with a **Canny edge filter** before being fed to the encoder. This removes appearance-level domain differences (lighting, seasons, sensor noise), keeping only structural/geometric information. Validated as the most impactful design choice in their ablation.

### Stage 1: Autoencoder pretraining

An autoencoder is trained to reconstruct Canny edge maps of satellite images:

```
L_AE = ||I - D(E(I))||²  +  β · Σ_l ||φ_l(I) - φ_l(D(E(I)))||²
```

- First term: pixel-wise L2 reconstruction on edge maps
- Second term: perceptual loss using DINOv2 features at multiple layers (β=1)

After training, **the decoder is discarded**. The encoder is kept as initialization for stage 2.

### Stage 2: VICRegL fine-tuning

The encoder is fine-tuned with VICRegL using **augmented satellite image pairs as positives**. The domain gap is bridged via augmentation simulating UAV characteristics:

| Augmentation | Parameters |
|---|---|
| Vignetting | Gaussian filter, σ=70 (most UAV-specific artifact) |
| Translation | Offsets within ±10m ground distance |
| Rotation | −30° to +30° |
| Cropping/Zoom | Scale 70–100% |
| Gaussian noise | Kernel size 5 |
| Blur | Gaussian, kernel size 5 |
| Brightness/Contrast | Factor 0–2 |

Loss: `L = α·VICReg(global) + (1−α)·[L_s + L_d]`, with α=0.75

Local matching uses two strategies simultaneously:
- **Location-based (L_s)**: match feature positions by spatial proximity, retaining top-γ=20 pairs
- **Feature-based (L_d)**: match each feature vector to its L2 nearest neighbor in the other view's embedding space

Projection heads (global: 4096-dim, 3 layers; local: 560-dim, 3 layers) are used during training and **discarded at inference**. Optimizer: AdamW, 100 epochs.

### Relevance to this project

`src/losses.py:SpatialLoss` already implements the VICRegL stage (both local matching strategies, VICReg variance/covariance terms). The main things CAEVL has that this project doesn't:

1. **Canny preprocessing** — their most impactful design choice
2. **Stage 1 autoencoder pretraining** — trains a fresh lightweight encoder; not applicable when using a frozen DiffusionSat backbone
3. **Vignetting augmentation** — not currently in `src/data_transforms.py:train_sat_uav_sim_transforms`
