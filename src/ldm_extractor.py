from dataclasses import dataclass, field

from archs.ldm_extractor import LDMExtractor


@dataclass
class LDMExtractorCfg:
  img_size: int = 256
  batch_size: int = 8
  save_timesteps: list = field(default_factory=lambda: [48, 46, 42])
  num_timesteps: int = 50
  layer_idxs: dict = field(default_factory=lambda: {"down_blocks": {"attn1": "all"}})
  diffusion_mode: str = "inversion"
  prompt: str = "A satellite image"
  negative_prompt: str = ""
  resize_outputs: int = -1
  max_i: int | None = None
  min_i: int | None = None

  def get(self, key, default=None):
    return getattr(self, key, default)
