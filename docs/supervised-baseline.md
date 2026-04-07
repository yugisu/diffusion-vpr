# Supervised Baseline

This goal of training a supervised baseline is to (a) verify the overall setup is valid, (b) define a supervised baseline to the embedding head trained using self-supervised approach.

## High-level model components

An image is passed into frozen DiffusionSatBackbone to produce a feature pyramid. The learnable `FuserEmbedder` head takes the feature pyramid as an input, and produces a linear embedding vector.

The model is converted to `bfloat16` (even though it might not have been trained with this precision, we want to minimize model footprint and speed up the backbone inference time). This could be changed?

### Frozen diffusion model based backbone

Frozen DiffusionSat-based backbone `DiffusionSatBackbone` (`backbone.py`) which takes an image as input, and produces a pyramid of spatial feature maps per DDIM inversion timestep per diffusion network's U-Net layer. Uses pytorch hooks for introspection inside the model during runtime.

Indexes for timesteps and U-Net layers are configurable hyperparameters. Given the experiments in `notebooks/2-evaluate-timesteps.py`, I found these values as optimal: `num_timesteps=10,save_timesteps=[8, 7],layer_idxs={'down_blocks': {'attn1': 'all'}}` (derived for zero-shot UAV-to-satellite retrieval). Comparison results can be found in `results/2-evaluate-timesteps.csv`, but all "good configurations" achieve near 2% R@1 for UAV to satellite retrieval (given it's zero-shot). Maybe a better way to find these out would be utilizing supervised training. SatDiFuser paper also gives good insight into which timesteps/layers to pick, and my findings more or less play around those recommendations.

Rough structure of the feature pyramid it produces:
```py
{
  "48": [
    Tensor(B, 640, 32, 32),   # down_blocks[0], 2x320ch concat
    Tensor(B, 1280, 16, 16),  # down_blocks[1], 2x640ch concat
    Tensor(B, 2560, 8, 8),    # down_blocks[2], 2x1280ch concat
  ],
  "46": [
    Tensor(B, 640, 32, 32),
    Tensor(B, 1280, 16, 16),
    Tensor(B, 2560, 8, 8),
  ],
  "42": [
    Tensor(B, 640, 32, 32),
    Tensor(B, 1280, 16, 16),
    Tensor(B, 2560, 8, 8),
  ],
}
```

### Embedding head

Trainable embedding head `FuserEmbedder` (`embedders.py`). Aggregates cross-timestep features from all specified layers, and produces a linear vector representing a specific image. Built on top of `GlobalWeightedFuser` from SatDiFuser.

`GlobalWeightedFuser` fuses the feature pyramid from the backbone to a list of shape:
```
scale_feats = [
  Tensor(B, 384, 32, 32),   # scale 32 — 3x2 contributions merged
  Tensor(B, 384, 16, 16),   # scale 16 — 3x2 contributions merged
  Tensor(B, 384,  8,  8),   # scale  8 — 3x2 contributions merged
]
```

Then we resize each entry of this list to the largest size, and pool it with GeM (advanced average pooling technique, basically), receiving an embedding tensor with shape `(B, 1152)`.

Basically, the head learns per-scale/per-timestep weights -> multi-scale spatial features resized to common HxW, concatenated -> GeM pooling -> L2 normalize -> final 384-dim embedding.

## Supervised training setup

### Data

Training data consists of pictures from 9 different flights shot on a Mavic-like drone in nadir, and a large satellite image of the location of the flight.

This dataset provides UAV images as standalone images, and satellite images in a form of a large satellite shot in TIFF format. So it's up to us to chunk the satellite images for further usage. We chunk the satellite imagery to loosely match UAV images' granularity, and produce 256x256 chunks with stride of `size//4` to ensure UAV shots have a close enough chunk with similar coordinates and similar ground coverage.

Training flights: `["01", "02", "04", "05", "06", "08", "09", "10", "11"]`, validation flight: `"03"`. The flights and a sample of the training data can be visually inspected in `notebooks/0-explore-visloc.ipynb`.

Augmentations for UAV/satellite imagery include affine transforms, rotations, crops, and color alterations, and can be found in `src/datasets/visloc.py`. The order of these transforms is crucial, one experiment has been failed due to crops before applying geometric transforms, introducing black corners in images and corrupting training samples.

### Training

The `FuserEmbedder` embedding head is trained on feature pyramids from UAV-satellite image pairs that went through the frozen `DiffusionSatBackbone` backbone.

After running the model, we receive two normalized embeddings per batch sample, one for the UAV image and for satellite image.

The embeddings batch is fed into an InfoNCE loss which promotes positive pairs over negative ones.

As for optimizer, AdamW is being used with a linear warmup + cosine decreasing learning rate schedule (min 1e-5, max 1e-2) and weight decay of 1e-4, total epochs capped at 50.

### Evaluation
R@{1,5,10} for retrieval of UAV image to satellite chunk. Validation is performed on flight 03, which is exempt from the training data. There is no designated test dataset, and the validation dataset is used in place of the testing dataset.
