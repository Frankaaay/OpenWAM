"""Frozen StereoTok codec using the existing FastWAM StereoTok vendor dependency."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from openwam.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from openwam.model.video_backbone.encoder.registry import register_video_encoder
from openwam.model.video_backbone.encoder.wan22_vae import _preprocess_image, _vae_output_to_image


@register_video_encoder("stereotok")
class StereoTokVideoEncoder(VideoEncoder):
    requires_stereo = True

    def __init__(self, codec, geometry):
        super().__init__()
        self._m = codec
        self.requires_stereo = codec.requires_stereo
        self.geometry = geometry
        self.train(False)

    @property
    def properties(self):
        return VideoEncoderProperties(48, 16, 4, True)

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    def preprocess_video(self, frames):
        parameter = next(self._m.parameters())
        return torch.stack([
            _preprocess_image(frame, dtype=torch.float32, device=parameter.device)
            for frame in frames
        ], dim=2)

    @torch.no_grad()
    def batch_encode(self, video, *, right_video=None):
        if video.shape[2] < 1 or (video.shape[2] - 1) % 4:
            raise ValueError("StereoTok requires 1+4k input frames; pad tails in the dataset")
        return self._m.encode(video, video.device, right_videos=right_video)

    def decode(self, latents, *, tiled=True):
        # The reused codec owns the supported untiled decode path.
        return self._m.decode(latents, latents.device, tiled=False)

    def to_frames(self, video_tensor):
        if video_tensor.shape[0] != 1:
            raise ValueError("to_frames requires one video")
        return [_vae_output_to_image(frame) for frame in video_tensor[0].permute(1, 2, 3, 0)]

    @classmethod
    def from_pretrained(cls, model_path: str, input_mode="stereo"):
        from fastwam.models.wan22.stereotok_vae import StereoTokVAE

        codec = StereoTokVAE(model_path, "cpu", input_mode=input_mode)
        # mmap reads only structural metadata here; the codec already strict-loaded weights.
        payload = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
        geometry = payload["config"]["geometry"] if payload["format"] == "stereotok-wan-training-v1" else payload["geometry"]
        return cls(codec, geometry)

    def save_deploy_assets(self, output_dir, cfg):
        metadata = {"geometry": self.geometry, "binding": self._m.binding}
        path = Path(output_dir) / "stereotok.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_skeleton(cls, components_entry, *, device="cpu", encoder_cfg=None, ckpt_dir=None):
        from fastwam.models.wan22.stereotok.checkpoints import DECODER_ARCHITECTURE
        from fastwam.models.wan22.stereotok.student import GeometryDecoder, LAS2GuidedFusion, make_vae
        from fastwam.models.wan22.stereotok_vae import StereoTokVAE

        if ckpt_dir is None:
            raise ValueError("StereoTok deploy requires checkpoint-local stereotok.json")
        metadata = json.loads((Path(ckpt_dir) / "stereotok.json").read_text(encoding="utf-8"))
        geometry = metadata["geometry"]
        if any(geometry.get(k) != v for k, v in DECODER_ARCHITECTURE.items()):
            raise ValueError("StereoTok deploy architecture mismatch")
        # Instantiate the same reused codec without reopening its training-time export.
        # Architecture.load_checkpoint supplies every persistent tensor by strict load.
        codec = StereoTokVAE.__new__(StereoTokVAE)
        torch.nn.Module.__init__(codec)
        codec.binding = metadata["binding"]
        input_mode = codec.binding.get("input_mode", "stereo")
        if input_mode not in ("stereo", "mono-left"):
            raise ValueError("Invalid StereoTok deployment input mode")
        if encoder_cfg is not None and encoder_cfg.get("input_mode", "stereo") != input_mode:
            raise ValueError("StereoTok deployment config and checkpoint input modes differ")
        codec.requires_stereo = input_mode == "stereo"
        with torch.device("meta"):
            model = make_vae()
            model.fusion = LAS2GuidedFusion(
                model.encoder, geometry["encoder_geometry"], geometry["max_disparity"], initialize=False
            )
            model.decoder.geometry_branch = GeometryDecoder(model.decoder)
        codec.model = model
        # Nonpersistent normalization buffers are not supplied by a checkpoint.
        codec.to_empty(device=device)
        matcher = model.fusion.matcher
        matcher.image_mean.copy_(torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1))
        matcher.image_std.copy_(torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1))
        return cls(codec, geometry)
