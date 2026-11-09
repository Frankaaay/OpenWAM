"""Small CPU contracts independent of optional trainer/model dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from fractions import Fraction
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
    return scope[name]


class Codec(nn.Linear):
    def __init__(self, channels=48):
        super().__init__(2, 2)
        self.properties = SimpleNamespace(z_dim=channels, spatial_compression=16,
            temporal_compression=4, causal_temporal=True, pixel_decode=True, dit_patch_size=(1, 2, 2))


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.swap = function('openwam/train/utils/ckpt_model_loader.py', '_replace_native_wan_codec', {})
        self.backbone = nn.Module()
        self.backbone.vae = nn.Linear(2, 2)
        self.backbone.video_encoder = None
        self.backbone.dit = nn.Module()
        self.backbone.dit.patch_size = (1, 2, 2)
        self.backbone.dit.patch_embedding = nn.Conv3d(48, 4, (1, 2, 2), stride=(1, 2, 2))
        self.backbone.dit.head = nn.Module()
        self.backbone.dit.head.patch_size = (1, 2, 2)
        self.backbone.dit.head.head = nn.Linear(4, 192)
        self.backbone.text_encoder = nn.Linear(2, 3)

    def test_swap_keeps_all_noncodec_parameter_objects_and_values(self):
        before = {k: (p, p.detach().clone()) for k, p in self.backbone.named_parameters()
                  if not k.startswith('vae.')}
        codec = Codec()
        self.swap(self.backbone, codec)
        after = dict(self.backbone.named_parameters())
        for k, (original, value) in before.items():
            self.assertIs(after[k], original)
            self.assertTrue(torch.equal(after[k], value))
        self.assertIsNone(self.backbone.vae)
        self.assertIs(self.backbone.video_encoder, codec)
        self.assertFalse(any(p.requires_grad for p in codec.parameters()))

    def test_incompatible_codec_fails_without_mutation(self):
        native = self.backbone.vae
        with self.assertRaises(ValueError):self.swap(self.backbone, Codec(16))
        self.assertIs(self.backbone.vae, native)
        self.assertIsNone(self.backbone.video_encoder)

    def test_incompatible_projection_fails_without_mutation(self):
        native = self.backbone.vae
        self.backbone.dit.head.head = nn.Linear(4, 16)
        with self.assertRaises(ValueError):self.swap(self.backbone, Codec())
        self.assertIs(self.backbone.vae, native)


class RightEyeTests(unittest.TestCase):
    def test_original_and_native_crop_keep_phase_pixels_and_tail_mask(self):
        camera='observation.images.cam_head_right'
        for native in (False,True):
            source_indices=np.arange(17)*3+2
            valid=np.array([True]*16+[False])
            frames=[]
            for i in range(53):
                arr=np.full((256,256,3),i,dtype=np.uint8)
                if native:arr=arr[48:208]
                frames.append(SimpleNamespace(pts=i,time_base=Fraction(1,30),
                    to_ndarray=lambda format,a=arr:a))
            stream=SimpleNamespace(time_base=Fraction(1,30),thread_count=0)
            container=SimpleNamespace(streams=SimpleNamespace(video=[stream]),
                seek=lambda *a,**kw:None,decode=lambda s:iter(frames))
            opened=[]
            def open_video(path):
                opened.append(path)
                return nullcontext(container)
            read=function('openwam/dataloader/orion_midtrain_ppu.py','_right_video',
                {'np':np,'Path':Path,'av':SimpleNamespace(open=open_video),'mosaic':lambda x:np.stack(x[0])})
            e={'availability/'+camera:native,'root':'native','episode_index':0,
               'native_source_episode_index':1005,'source_split':'train'}
            locator={'chunk_index':1,'file_index':5,'from_timestamp':0.0}
            e.update({'videos/'+camera+'/'+k:v for k,v in locator.items()})
            root='/gpfs/jiuquyun/datasets/PRETRAIN_DATA/ABC-130k-yam-canonical-v1/'
            owner=SimpleNamespace(right_video_root=Path('repaired'),right_repairs={1005:{
                'source_split':'train','locator':locator,
                'verification':{'path':root+'videos/'+camera+'/chunk-001/file-005.mp4'}}})
            got=read(owner,e,{'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'},
                {'crops':{camera:[0,48,256,160],'observation.images.cam_head_left':[0,48,256,160]}},
                SimpleNamespace(frame_valid=valid),source_indices,source_indices/30.)
            expected=np.zeros((17,256,256,3),dtype=np.uint8)
            for i,index in enumerate(source_indices[:-1]):expected[i,48:208]=index
            np.testing.assert_array_equal(got,expected)
            self.assertIn('file-005.mp4',opened[0])

    def test_right_origin_mismatch_rejected_before_video_read(self):
        read = function('openwam/dataloader/orion_midtrain_ppu.py', '_right_video', {'np':np, 'Path':Path})
        camera = 'observation.images.cam_head_right'
        e = {'availability/'+camera:True, 'root':'unused'}
        e.update({'videos/'+camera+'/'+k:v for k,v in
                  {'chunk_index':0,'file_index':0,'from_timestamp':1.0}.items()})
        with self.assertRaisesRegex(ValueError, 'timestamp origin'):
            read(SimpleNamespace(),e,{'video_path':'{video_key}/{file_index}.mp4'},
                 {'crops':{camera:[0,48,256,160]}},None,np.arange(17)*3,np.arange(17)/10)


if __name__ == '__main__':unittest.main()
