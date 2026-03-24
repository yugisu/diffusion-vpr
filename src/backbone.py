from pathlib import Path

import torch
from diffusers import AutoencoderKL
from diffusionsat import DiffusionSatPipeline, SatUNet

from src.ldm_extractor import LDMExtractor, LDMExtractorCfg

SD_REVISION = None


class DiffusionSatBackbone(torch.nn.Module):
  def __init__(self, checkpoint_path: str | Path, device: torch.device):
    super().__init__()

    self.checkpoint_path = checkpoint_path
    self.device = device

    # VAE
    self.vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae", revision=SD_REVISION)
    self.vae.requires_grad_(False)
    self.vae.to(device)

    # Diffusion UNet and pipeline.
    unet = SatUNet.from_pretrained(
      checkpoint_path,
      subfolder="checkpoint-150000/unet",
      revision=SD_REVISION,
      num_metadata=0,
      use_metadata=False,
      low_cpu_mem_usage=False,
    )
    unet.requires_grad_(False)
    unet.to(device)

    self.pipe = DiffusionSatPipeline.from_pretrained(checkpoint_path, unet=unet, low_cpu_mem_usage=False)
    self.pipe.to(device)

    # Diffusion feature extractor from SatDiFuser. Must be configured separately since it requires hyperparameters and to avoid re-loading the UNet.
    self.ldm_extractor: LDMExtractor | None = None

  def set_ldm_extractor_cfg(self, cfg: LDMExtractorCfg):
    self.ldm_extractor_cfg = cfg
    self.ldm_extractor = LDMExtractor(cfg, self.pipe)

  @torch.no_grad()
  def forward(self, imgs: torch.Tensor) -> dict:
    """
    Returns a dict of features from the configured LDM extractor.
    """

    if self.ldm_extractor is None:
      raise ValueError("LDM extractor not configured. Please call set_ldm_extractor_cfg() first.")

    latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215
    feats, _ = self.ldm_extractor.forward(latents)
    return feats
