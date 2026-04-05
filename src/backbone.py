from pathlib import Path

import torch
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusionsat import DiffusionSatPipeline, SatUNet

from src.ldm_extractor import LDMExtractor, LDMExtractorCfg

SD_REVISION = None


class DiffusionSatBackbone(torch.nn.Module):
  def __init__(self, checkpoint_path: str | Path, device: torch.device, dtype: torch.dtype):
    super().__init__()

    self.checkpoint_path = checkpoint_path
    self.device = device
    self.dtype = dtype

    # VAE
    self.vae = AutoencoderKL.from_pretrained(
      checkpoint_path,
      subfolder="vae",
      revision=SD_REVISION,
      torch_dtype=self.dtype,
    )
    self.vae.requires_grad_(False)
    self.vae.to(device)

    # Diffusion UNet and pipeline.
    self.unet = SatUNet.from_pretrained(
      checkpoint_path,
      subfolder="checkpoint-150000/unet",
      revision=SD_REVISION,
      num_metadata=0,
      use_metadata=False,
      low_cpu_mem_usage=False,
      torch_dtype=self.dtype,
    )
    self.unet.requires_grad_(False)
    self.unet.set_attn_processor(AttnProcessor2_0())  # NOTE/PERF: SDPA
    self.unet.to(device)

    self.unet.to(memory_format=torch.channels_last)
    self.vae.to(memory_format=torch.channels_last)

    self.pipe = DiffusionSatPipeline.from_pretrained(
      checkpoint_path,
      vae=self.vae,
      unet=self.unet,
      low_cpu_mem_usage=False,
      torch_dtype=self.dtype,
    )
    self.pipe.to(device)

    # Diffusion feature extractor from SatDiFuser. Must be configured separately since it requires hyperparameters and to avoid re-loading the UNet.
    self.ldm_extractor: LDMExtractor | None = None

  def set_ldm_extractor_cfg(self, cfg: LDMExtractorCfg):
    self.ldm_extractor_cfg = cfg
    with torch.autocast(str(self.device), dtype=self.dtype):
      self.ldm_extractor = LDMExtractor(cfg, self.pipe)

  @torch.inference_mode()
  def forward(self, imgs: torch.Tensor) -> dict:
    """
    Returns a dict of features from the configured LDM extractor.
    """

    if self.ldm_extractor is None:
      raise ValueError("LDM extractor not configured. Please call set_ldm_extractor_cfg() first.")

    latents = self.vae.encode(imgs.to(dtype=self.dtype)).latent_dist.sample() * 0.18215
    feats, _ = self.ldm_extractor.forward(latents)
    return feats
