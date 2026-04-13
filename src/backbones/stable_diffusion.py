import torch
import torch.nn as nn
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

SD21 = "sd2-community/stable-diffusion-2-1"


class SD21Backbone(torch.nn.Module):
  def __init__(
    self,
    layers: list[str],
    timestep: int,
    prompt: str = "a satellite image",
  ):
    super().__init__()

    self.layers = layers
    self.timestep = timestep

    # Diffusion UNet and pipeline.
    self.unet = UNet2DConditionModel.from_pretrained(
      SD21,
      subfolder="unet",
      low_cpu_mem_usage=False,
    )
    self.unet.requires_grad_(False)

    self.pipe = StableDiffusionPipeline.from_pretrained(
      SD21,
      unet=self.unet,
      low_cpu_mem_usage=False,
      torch_dtype=self.dtype,
    )
    self.pipe.to("cuda")
    self.pipe.enable_attention_slicing()
    self.pipe.enable_xformers_memory_efficient_attention()

    self._prompt_embeds = self.pipe._encode_prompt(
      prompt=prompt,
      device="cuda",
      num_images_per_prompt=1,
      do_classifier_free_guidance=False,
    )

    self._features: dict[str, torch.Tensor] = {}
    self._handles = []
    for name in layers:
      module = self.pipe.unet.get_submodule(name)
      self._handles.append(module.register_forward_hook(self._make_hook(name)))

  def _make_hook(self, name: str):
    def hook(_module, _inputs, output):
      self._features[name] = output[0] if isinstance(output, tuple) else output
      self._features[name].detach()

    return hook

  @torch.inference_mode()
  def forward(
    self,
    imgs: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    latents = self.pipe.vae.encode(imgs).latent_dist.sample() * 0.18215

    t = torch.tensor(self.timestep, device=latents.device, dtype=torch.long)
    noise = torch.randn_like(latents).to(latents.device)
    latents_noisy = self.scheduler.add_noise(latents, noise, t)

    prompt_embeds = self._prompt_embeds.repeat(latents.shape[0], 1, 1)

    self._features.clear()

    _ = self.pipe(
      img_tensor=latents_noisy,
      t=t,
      prompt_embeds=prompt_embeds,
    )

    return dict(self._features)
