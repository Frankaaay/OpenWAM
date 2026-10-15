"""Real u8000 -> pretrained Wan forward; no trainer or optimizer is constructed."""
import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.transforms.multiview import assemble_multiview_layout, format_prompt_for_inference
from openwam.model.video_backbone.encoder import build_video_encoder
from openwam.model.video_backbone.registry import build_video_backbone


@torch.no_grad()
def probe(weights, student, data_root, manifest_path):
    start = time.time()
    encoder = build_video_encoder({'name': 'stereotok', 'model_path': student}).to('cuda')
    backbone = build_video_backbone(
        'wan22_ti2v_5b', {'video_backbone': {'from_scratch': False}},
        source={'model_path': weights}, external_encoder=encoder, device='cuda',
    )
    backbone.eval().requires_grad_(False)
    print('REAL_WAN_FACTORY_LOADED', flush=True)
    episode = json.loads(Path(manifest_path).read_text())['episodes'][0]
    clips = [[], []]
    cameras = ['head_camera', 'left_camera', 'right_camera']
    with h5py.File(Path(data_root) / episode['path']) as handle:
        for eye, suffix in enumerate(['', '_right']):
            for index in range(0, 33, 4):
                images = {}
                for camera in cameras:
                    raw = handle[f'observation/{camera}{suffix}/rgb'][index]
                    pixels = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
                    if pixels is None:
                        raise ValueError(f'Undecodable image: {camera}{suffix}/{index}')
                    images[camera] = Image.fromarray(pixels)
                clips[eye].append(assemble_multiview_layout(images, cameras, 384, 320))
    prompt = format_prompt_for_inference(episode['seen_instructions'][0])
    prepared = backbone.preprocess_input_for_train(frames=[clips[0]], right_frames=[clips[1]], text=[prompt])
    state = backbone.prepare(latents=prepared['input_latents'],
        timestep=torch.tensor([500.], device='cuda'), context=prepared['context'],
        seq_lens=prepared['seq_lens'], fuse_vae_embedding_in_latents=True)
    for layer in range(backbone.num_layers):
        state = backbone.run_block(layer, state)
    prediction = backbone.finalize(state)
    assert torch.isfinite(prediction).all()
    first = backbone.preprocess_input_for_inference(
        prompt=prompt, first_frame_image=clips[0][0], right_first_frame_image=clips[1][0],
        num_frames=9, height=384, width=320, num_inference_steps=1,
    )
    error = (first['first_frame_latents'] - prepared['input_latents'][:, :, :1]).abs().max().item()
    assert error == 0, error
    return dict(production_factory=True, latent_shape=list(prepared['input_latents'].shape),
                prediction_shape=list(prediction.shape), finite=True, first_frame_max_abs=error,
                dit_layers=len(backbone.dit.blocks), real_robotwin=True, action_branch_tested=False,
                seconds=time.time() - start, training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['weights', 'student', 'data-root', 'manifest', 'output']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = probe(args.weights, args.student, args.data_root, args.manifest)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
