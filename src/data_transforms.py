from torchvision import transforms

# Satellite view: light augmentation — color jitter only, no geometry.
train_sat_transforms = transforms.Compose(
  [
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)

# UAV-simulated view: heavy augmentation to bridge the domain gap.
# Perspective warp models oblique angle; strong color jitter + blur model
# sensor and lighting differences; rotation models non-nadir heading.
train_sat_uav_sim_transforms = transforms.Compose(
  [
    transforms.RandomPerspective(distortion_scale=0.4, p=0.8),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.6, 0.6, 0.6, 0.2)], p=0.9),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply(
      [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)


inference_transforms = transforms.Compose(
  [
    transforms.Resize(
      (256, 256),
      interpolation=transforms.InterpolationMode.BILINEAR,
      antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)
