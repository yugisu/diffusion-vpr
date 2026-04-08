# Image augmentations research

This doc captures which image augmentations different architectures use.

### SatMAE

https://github.com/sustainlab-group/SatMAE/blob/main/util/datasets.py#L50

Satellite only images.

```py
t = []
if is_train:
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    t.append(
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
    )
    t.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(t)
```

### CAEVL

https://github.com/Tristan-Amadei/caevl/tree/main/caevl/data_generation/dataset/augmentations

Clean satellite
```
transforms.Compose([
    Resize((256, 256)),
    Grayscale(),                         # → 1-channel grayscale
    # if canny_edges=True:
    CannyMask(low_threshold=70,
              high_threshold=200),       # cv2.Canny → binary {0,255} edge map
    ToTensor(),                          # PIL/ndarray → float32 [0, 1]
    # if apply_normalization=True:
    Normalize(mean, std),
])
```

UAV view simulation:
```
transforms.Compose([
    Resize((256, 256)),
    Grayscale(),
    # p=0.8 — randomly applies one of three geometric distortions:
    RandomZoomIn(degrees=30, translate=(50, 50), p=0.8)  # picks uniformly from:
    #   (a) p≈1/3: rotate ±30°, center-crop to largest inscribed rect, resize back
    #   (b) p≈1/3: affine shift ±50px (each axis p=0.75), crop non-zero bbox, resize back
    #   (c) p≈1/3: RandomResizedCrop(scale=(0.7,1.0), ratio=1:1), resize back

    RandomGaussianBlur(kernel_size=5, p=0.5),

    RandomBrightnessContrast(
        contrast_factor=1.0,   contrast_p=0.5,   # scale by uniform(0.1, 2.0)
        brightness_factor=1.0, brightness_p=0.5, # scale by uniform(1.0, 2.0)
    ),

    RandomGaussianNoise(mean=0, std=1, p=0.25, clip=True),  # N(0,1) additive noise

    RandomVignetting(sigma=70, p=0.9),  # Gaussian falloff mask; stronger darkening
                                        # inside a central circle (sigma/1.5 there)

    # if canny_edges=True:
    CannyMask(low_threshold=70, high_threshold=200),

    ToTensor(),

    RandomMasking(p=1.0),  # always: multiply by a fixed binary_mask.npy
                           # (a pre-saved shape mask, likely a UAV camera frame/silhouette)

    # if apply_normalization=True:
    Normalize(mean, std),
])
```

Inference UAV transform:
```
transforms.Compose([
    Resize((256, 256)),
    Grayscale(),
    # if canny_edges=True:
    CannyMask(low_threshold=70, high_threshold=200),
    ToTensor(),
    # if apply_normalization=True:
    Normalize(mean, std),
])

```

### C^2FFViT (Trim-UAV-VisLoc paper)

https://www.mdpi.com/2072-4292/17/17/3045

to the raw UAV imagery. Specifically, each image was rotated by 90° and 180°, subjected to random scaling (both up-sampling and down-sampling), and adjusted in contrast to emulate varying illumination conditions. For every augmented sample, we re-computed the Intersection over Union (IoU) between predicted and ground-truth bounding boxes and then incorporated these augmented examples into the training set for a second round of model optimization.

### Other

https://claude.ai/chat/8018624b-dd9c-49fa-90e4-b55c7b26d716

Depend on geometry transforms, mostly.
