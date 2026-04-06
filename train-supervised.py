from src.model import FuserEmbedderValidationMixin

print("Running train-supervised.py...")
import math
import os
import random
import warnings
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from dotenv import load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.backbone import DiffusionSatBackbone
from src.datasets.visloc import (
  SatChunkDataset,
  UAVDataset,
  inference_sat_transforms,
  inference_uav_transforms,
)
from src.embedders import FuserEmbedder
from src.ldm_extractor import LDMExtractorCfg

print("Collecting .env...")

load_dotenv()

VISLOC_ROOT = Path(os.environ["VISLOC_ROOT"])
DIFFUSIONSAT_256_CHCKPT = Path(os.environ["DIFFUSIONSAT_256_CHCKPT"])
RANDOM_SEED = 42
DEVICE = torch.device("cuda")
NUM_WORKERS = 8
BATCH_SIZE = 256
FLIGHT_IDS = ["01", "02", "04", "05", "06", "08", "09", "10", "11"]
VAL_FLIGHT_ID = "03"

L.seed_everything(RANDOM_SEED, workers=True)
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# ---------------------------------------------------------------------------
# UAV training augmentations
# ---------------------------------------------------------------------------

_random_90 = transforms.Lambda(lambda img: TF.rotate(img, random.choice([0, 90, 180, 270])))

train_uav_transforms = transforms.Compose(
  [
    # Geometric
    transforms.Resize(300),
    transforms.RandomResizedCrop(256, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
    _random_90,
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.4),
    # Photometric — UAV images often have different exposure, haze, and sensor response
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)], p=0.8),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    # Simulate motion blur from drone movement
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(1, 7), sigma=(0.1, 1.0))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)

# Satellite augmentation — heavy, symmetric with UAV side
train_sat_transforms = transforms.Compose(
  [
    # Geometric — satellite tiles can appear at any orientation and scale
    transforms.Resize(300),
    transforms.RandomResizedCrop(256, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
    _random_90,
    # Photometric — sensor/season/time-of-day variation
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)], p=0.8),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ]
)


# ---------------------------------------------------------------------------
# Paired dataset
# ---------------------------------------------------------------------------


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
  ):
    self.uav_ds = UAVDataset(root, flight_id, transform=None)
    self.sat_ds = SatChunkDataset(
      root,
      flight_id=flight_id,
      chunk_pixels=512,
      stride_pixels=128,
      scale_factor=0.25,
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


# ---------------------------------------------------------------------------
# Supervised model module
# ---------------------------------------------------------------------------


class SupervisedEmbedderModule(FuserEmbedderValidationMixin, L.LightningModule):
  """Trains FuserEmbedder with cross-modal InfoNCE loss on (UAV, satellite) pairs."""

  def __init__(
    self,
    backbone: DiffusionSatBackbone,
    # General
    img_size: Literal[256] = 256,
    batch_size: int = 16,
    # LDM extractor
    save_timesteps: list = [8, 7],
    num_timesteps: int = 10,
    layer_idxs: dict = {"down_blocks": {"attn1": "all"}},
    diffusion_mode: str = "inversion",
    prompt: str = "A satellite image",
    negative_prompt: str = "",
    resize_outputs: int = -1,
    # Loss
    temperature: float = 0.07,
    # Training
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_epochs: int = 5,
    max_train_epochs: int = 40,
    # Validation
    val_gallery_dataloader: torch.utils.data.DataLoader | None = None,
  ):
    super().__init__()
    self.save_hyperparameters(ignore=["backbone", "val_gallery_dataloader"])

    ldm_cfg = LDMExtractorCfg(img_size=img_size, batch_size=batch_size, save_timesteps=save_timesteps, num_timesteps=num_timesteps, layer_idxs=layer_idxs, diffusion_mode=diffusion_mode, prompt=prompt, negative_prompt=negative_prompt, resize_outputs=resize_outputs)  # fmt: off
    backbone.set_ldm_extractor_cfg(ldm_cfg)
    self.backbone = backbone
    self.backbone.eval()

    self.embedder = FuserEmbedder(
      save_timesteps=self.backbone.ldm_extractor.save_timesteps,
      feature_dims=self.backbone.ldm_extractor.collected_dims,
    )

    self.temperature = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    self.val_gallery_dataloader = val_gallery_dataloader

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  @torch.inference_mode()
  def forward(self, imgs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    feats = self.backbone(imgs)

    embs = self.embedder.forward(feats)
    if normalize:
      embs = F.normalize(embs, p=2, dim=1)

    return embs

  def _embed_train(self, imgs: torch.Tensor) -> torch.Tensor:
    """Feature extraction (no_grad) + trainable head (with grad)."""
    with torch.no_grad():
      feats = self.backbone(imgs)
    embs = self.embedder(feats)
    return F.normalize(embs, p=2, dim=1)

  def _infonce_loss(self, uav_embs: torch.Tensor, sat_embs: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style) loss. Both directions averaged."""
    logit_scale = self.temperature.exp().clamp(max=100)
    logits = logit_scale * uav_embs @ sat_embs.T  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_uav = F.cross_entropy(logits, labels)
    loss_sat = F.cross_entropy(logits.T, labels)
    return (loss_uav + loss_sat) / 2

  # ------------------------------------------------------------------
  # Training
  # ------------------------------------------------------------------

  def training_step(self, batch, _batch_idx):
    uav_imgs, sat_imgs, _, _ = batch
    uav_embs = self._embed_train(uav_imgs)
    sat_embs = self._embed_train(sat_imgs)
    loss = self._infonce_loss(uav_embs, sat_embs)
    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log("train/temperature", self.temperature.exp(), on_step=True)
    return loss

  # ------------------------------------------------------------------
  # Optimizer
  # ------------------------------------------------------------------

  def configure_optimizers(self):
    hp = self.hparams
    params = list(self.embedder.parameters()) + [self.temperature]
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)

    total_steps = self.trainer.estimated_stepping_batches
    steps_per_epoch = max(1, total_steps // hp.max_train_epochs)
    warmup_steps = hp.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
      if step < warmup_steps:
        return step / max(1, warmup_steps)
      t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
      return max(hp.min_lr / hp.lr, 0.5 * (1.0 + math.cos(math.pi * t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ---------------------------------------------------------------------------
# Datasets & dataloaders
# ---------------------------------------------------------------------------

print("Setting up training dataset...")

from torch.utils.data import ConcatDataset

train_ds = ConcatDataset(
  [
    PairedUAVSatDataset(
      VISLOC_ROOT, flight_id=fid, uav_transform=train_uav_transforms, sat_transform=train_sat_transforms
    )
    for fid in FLIGHT_IDS
  ]
)
train_loader = DataLoader(
  train_ds,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
  shuffle=True,
  drop_last=True,
)

print(f"Training pairs: {len(train_ds)} across {len(FLIGHT_IDS)} flights")

print("Setting up validation datasets...")


val_query_ds = UAVDataset(VISLOC_ROOT, flight_id=VAL_FLIGHT_ID, transform=inference_uav_transforms)
val_query_loader = DataLoader(val_query_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

val_gallery_ds = SatChunkDataset(
  VISLOC_ROOT,
  flight_id=VAL_FLIGHT_ID,
  chunk_pixels=512,
  stride_pixels=128,
  scale_factor=0.25,
  transform=inference_sat_transforms,
)
val_gallery_loader = DataLoader(val_gallery_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MAX_EPOCHS = 50

print("Setting up backbone...")

backbone = DiffusionSatBackbone(DIFFUSIONSAT_256_CHCKPT, DEVICE, dtype=torch.bfloat16)

print("Setting up model...")

model = SupervisedEmbedderModule(
  backbone=backbone,
  img_size=256,
  batch_size=BATCH_SIZE,
  max_train_epochs=MAX_EPOCHS,
  temperature=0.07,
  save_timesteps=[8, 7],
  num_timesteps=10,
  layer_idxs={"down_blocks": {"attn1": "all"}},
  val_gallery_dataloader=val_gallery_loader,
  warmup_epochs=2,
  lr=1e-2,
  weight_decay=1e-4,
  min_lr=1e-4,
)

print("Model hparams:")
for name, param in model.hparams.items():
  print(f"  {name}: {param}")


# ---------------------------------------------------------------------------
# Logger & callbacks
# ---------------------------------------------------------------------------

print("Setting up logger and callbacks...")

wandb_logger = WandbLogger(project="diffusion-vpr", log_model=False)
wandb_logger.log_hyperparams(
  {
    "dataset": "visloc",
    "train_flights": FLIGHT_IDS,
    "val_flights": [VAL_FLIGHT_ID],
  }
)
wandb_logger.watch(model.embedder, log="gradients", log_freq=100)

checkpoint_cb = ModelCheckpoint(
  dirpath="checkpoints",
  filename="supervised-{epoch:02d}-{val/Recall@1:.4f}",
  monitor="val/Recall@1",
  mode="max",
  save_top_k=3,
  save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval="step")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print("Setting up trainer...")

trainer = L.Trainer(
  max_epochs=MAX_EPOCHS,
  accelerator="gpu",
  devices=1,
  precision="bf16-mixed",
  logger=wandb_logger,
  callbacks=[checkpoint_cb, lr_monitor],
  val_check_interval=1.0,
  log_every_n_steps=5,
  gradient_clip_val=1.0,
)

print("Starting training!")

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_query_loader)
