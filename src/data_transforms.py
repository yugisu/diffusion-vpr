from torchvision import transforms

# Satellite view: light augmentation — color jitter only, no geometry.
train_sat_transforms = transforms.Compose(
  [
    transforms.Resize((256, 256)),
    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)


# Not exactly CAEVL-style augmentation, but aims to simulate a similar domain gap shift.
train_sat_uav_sim_transforms = transforms.Compose(
  [
    transforms.RandomPerspective(distortion_scale=0.2, p=0.0),  # nadir only
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),  # this can ruin the orientation?
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(384),
    transforms.Resize((256, 256)),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.9),
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
