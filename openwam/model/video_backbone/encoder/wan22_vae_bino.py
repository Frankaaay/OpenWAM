"""Naive binocular Wan baseline: the frozen Wan2.2 VAE encodes each eye; latents are joined along width."""

from __future__ import annotations

import torch
from torch import Tensor

from openwam.model.video_backbone.encoder.registry import register_video_encoder
from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder


@register_video_encoder("wan22_vae_bino")
class WanBinoVideoEncoder(WanVideoVAEEncoder):
    requires_stereo = True
    width_factor = 2

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    @torch.no_grad()
    def batch_encode(self, video: Tensor, *, right_video: Tensor | None = None) -> Tensor:
        if right_video is None or right_video.shape != video.shape:
            raise ValueError("Wan binocular baseline requires a matching right-eye tensor")
        # Same native Wan call as the monocular baseline, once per eye.
        return torch.cat([self._m.batch_encode(video, device=video.device),
                          self._m.batch_encode(right_video, device=right_video.device)], dim=-1)

    def decode(self, latents: Tensor, *, tiled: bool = True) -> Tensor:
        left, right = latents.chunk(2, dim=-1)
        return torch.cat([super().decode(left, tiled=tiled), super().decode(right, tiled=tiled)], dim=-1)


__all__ = ["WanBinoVideoEncoder"]
