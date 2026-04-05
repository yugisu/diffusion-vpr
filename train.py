print("Running train.py...")
import os
import warnings
from pathlib import Path

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

from src.backbone import DiffusionSatBackbone
from src.datasets.visloc import (
  SatChunkDataset,
  UAVDataset,
  inference_sat_transforms,
  inference_uav_transforms,
  train_sat_uav_sim_transforms,
)
from src.model import FuserEmbedderModule

print("Collecting .env...")

# Load .env BEFORE accessing env vars.
load_dotenv()


VISLOC_ROOT = Path(os.environ["VISLOC_ROOT"])
DIFFUSIONSAT_256_CHCKPT = Path(os.environ["DIFFUSIONSAT_256_CHCKPT"])
RANDOM_SEED = 42
DEVICE = torch.device("cuda")
NUM_WORKERS = 8
BATCH_SIZE = 256
MAX_EPOCHS = 10
TRAIN_FLIGHT_IDS = ["03"] # overfit to validate
# TRAIN_FLIGHT_IDS = ["01", "02", "05", "09", "11"]
VAL_FLIGHT_ID = "03"

L.seed_everything(RANDOM_SEED, workers=True)
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# ---------------------------------------------------------------------------
# Training dataloader
# ---------------------------------------------------------------------------

print("Setting up training datasets...")

def train_collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Apply two independent UAV-sim augmentations per satellite chunk to produce a pair of views."""
  views_a, views_b, lats, lons = [], [], [], []
  for img, lat, lon in batch:
    views_a.append(train_sat_uav_sim_transforms(img))
    views_b.append(train_sat_uav_sim_transforms(img))
    lats.append(lat)
    lons.append(lon)
  return torch.stack(views_a), torch.stack(views_b), torch.tensor(lats), torch.tensor(lons)


train_ds = ConcatDataset(
  [
    SatChunkDataset(VISLOC_ROOT, flight_id=fid, chunk_pixels=512, stride_pixels=128, scale_factor=0.25)
    for fid in TRAIN_FLIGHT_IDS
  ]
)
train_loader = DataLoader(
  train_ds,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
  shuffle=True,
  drop_last=True,
  collate_fn=train_collate_fn,
)


# ---------------------------------------------------------------------------
# Validation dataloaders
# ---------------------------------------------------------------------------

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
val_gallery_loader = DataLoader(
  val_gallery_ds,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
  shuffle=False,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print("Setting up backbone...")

backbone = DiffusionSatBackbone(DIFFUSIONSAT_256_CHCKPT, DEVICE, dtype=torch.bfloat16)

print("Setting up model...")

model = FuserEmbedderModule(
  backbone=backbone,
  img_size=256,
  batch_size=BATCH_SIZE,
  val_batch_size=BATCH_SIZE,
  warmup_epochs=1,
  max_train_epochs=MAX_EPOCHS,
  vicreg_alpha=0.5,
  save_timesteps=[48, 46, 42],
  num_timesteps=50,
  layer_idxs={"down_blocks": {"attn1": "all"}},
  val_gallery_dataloader=val_gallery_loader,
)

print("Model hparams:")
for name, param in model.hparams.items():
    print(f"  {name}: {param}")


# ---------------------------------------------------------------------------
# Logger & callbacks
# ---------------------------------------------------------------------------

print("Setting up logger and callbacks...")

wandb_logger = WandbLogger(
  project="diffusion-vpr",
  log_model=False,
)
wandb_logger.watch(model.embedder, log="gradients", log_freq=100)

checkpoint_cb = ModelCheckpoint(
  dirpath="checkpoints",
  filename="fuser-{epoch:02d}-{val/Recall@1:.4f}",
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
  log_every_n_steps=10,
  gradient_clip_val=1.0,
)

print("Starting training!")

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_query_loader)
