import math
from typing import Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from src.backbone import DiffusionSatBackbone
from src.embedders import FuserEmbedder
from src.evaluation import calculate_metrics, plot_tsne
from src.ldm_extractor import LDMExtractorCfg
from src.losses import SpatialLoss
from src.retrievers import FAISSRetriever


class FuserEmbedderValidationMixin:
  """Mixin to add validation logic for the FuserEmbedderModule."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Validation state.
    self._val_gallery_embs: list[torch.Tensor] | None = None
    self._val_query_embs: list[torch.Tensor] | None = None
    self._val_gallery_coords: list[tuple[float, float]] | None = None
    self._val_query_coords: list[tuple[float, float]] | None = None

  # ------------------------------------------------------------------
  # Validation
  # ------------------------------------------------------------------

  @torch.inference_mode()
  def on_validation_epoch_start(self):
    if self.val_gallery_dataloader is None:
      return

    # Build the satellite database before any validation_step runs.
    self._val_gallery_embs = []
    self._val_gallery_coords = []

    for imgs, lats, lons in tqdm(self.val_gallery_dataloader, desc="Building gallery embeddings"):
      embs = self.forward(imgs.to(self.device, dtype=self.dtype)).cpu()
      self._val_gallery_embs.extend(embs)
      self._val_gallery_coords.extend(zip(lats.tolist(), lons.tolist()))

    self._val_gallery_retriever = FAISSRetriever(torch.stack(self._val_gallery_embs))

    self._val_query_embs = []
    self._val_query_coords = []

  @torch.inference_mode()
  def validation_step(self, batch, _batch_idx):
    if self.val_gallery_dataloader is None:
      return

    imgs, lats, lons = batch
    embs = self.forward(imgs.to(self.device, dtype=self.dtype)).cpu()

    self._val_query_embs.extend(embs)
    self._val_query_coords.extend(zip(lats.tolist(), lons.tolist()))

  @torch.inference_mode()
  def on_validation_epoch_end(self):
    if self.val_gallery_dataloader is None or not self._val_query_embs or not self._val_gallery_embs:
      return

    query_embs = torch.stack(self._val_query_embs)
    _, preds = self._val_gallery_retriever.search(query_embs, k=10)

    # Ground truth: for each query, the single closest gallery image by GPS distance.
    gallery_coords = np.array(self._val_gallery_coords)  # (N, 2)
    query_coords = np.array(self._val_query_coords)  # (M, 2)
    dists = np.sum((query_coords[:, None, :] - gallery_coords[None, :, :]) ** 2, axis=2)
    gt = [[int(np.argmin(d))] for d in dists]

    metrics = calculate_metrics(preds, gt)
    for name, value in metrics.items():
      self.log(f"val/{name}", value, prog_bar=True)

    # Log t-SNE visualization of paired query/gallery embeddings.
    gallery_embs = torch.stack(self._val_gallery_embs)
    gt_indices = [g[0] for g in gt]
    matched_gallery = gallery_embs[gt_indices]
    fig = plot_tsne(matched_gallery.numpy(), query_embs.numpy())
    if self.logger:
      self.logger.experiment.log({"val/tsne": wandb.Image(fig)}, step=self.global_step)
    plt.close(fig)

    # Clear validation state.
    self._val_gallery_retriever = None
    self._val_gallery_embs = None
    self._val_gallery_coords = None
    self._val_query_embs = None
    self._val_query_coords = None


class FuserEmbedderModule(FuserEmbedderValidationMixin, L.LightningModule):
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
    vicreg_alpha: float = 0.5,
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

    # Diffusion backbone.
    ldm_cfg = LDMExtractorCfg(img_size=img_size, batch_size=batch_size, save_timesteps=save_timesteps, num_timesteps=num_timesteps, layer_idxs=layer_idxs, diffusion_mode=diffusion_mode, prompt=prompt, negative_prompt=negative_prompt, resize_outputs=resize_outputs)  # fmt: off
    backbone.set_ldm_extractor_cfg(ldm_cfg)
    self.backbone = backbone
    self.backbone.eval()

    # Trainable head.
    self.embedder = FuserEmbedder(
      save_timesteps=self.backbone.ldm_extractor.save_timesteps,
      feature_dims=self.backbone.ldm_extractor.collected_dims,
    )

    self.criterion = SpatialLoss(alpha=vicreg_alpha)

    # Validation dataset.
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

  # ------------------------------------------------------------------
  # Training
  # ------------------------------------------------------------------

  def training_step(self, batch, _batch_idx):
    sat_view, uav_view, _, _ = batch

    with torch.no_grad():
      feats1 = self.backbone(sat_view)
      feats2 = self.backbone(uav_view)

    spatial1, pooled1 = self.embedder.forward_unpooled(feats1)
    spatial2, pooled2 = self.embedder.forward_unpooled(feats2)

    loss = self.criterion(spatial1, pooled1, spatial2, pooled2)
    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    return loss

  # ------------------------------------------------------------------
  # Optimizer
  # ------------------------------------------------------------------

  def configure_optimizers(self):
    hp = self.hparams
    optimizer = torch.optim.AdamW(
      self.embedder.parameters(),
      lr=hp.lr,
      weight_decay=hp.weight_decay,
    )

    total_steps = self.trainer.estimated_stepping_batches
    steps_per_epoch = total_steps // hp.max_train_epochs
    warmup_steps = hp.warmup_epochs * steps_per_epoch

    # Cosine decay with linear warmup.
    def lr_lambda(step: int) -> float:
      if step < warmup_steps:
        return step / max(1, warmup_steps)
      t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
      return max(
        hp.min_lr / hp.lr,
        0.5 * (1.0 + math.cos(math.pi * t)),
      )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return {
      "optimizer": optimizer,
      "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }
