from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.registry import DATASET_REGISTRY


def test_stereo_decode_preserves_left_layout_and_requires_real_right(monkeypatch):
    from PIL import Image
    reader = object.__new__(LiberoDataset)
    reader.stereo = True
    reader._head_camera = 'head'
    reader._left_wrist_camera = 'wrist'
    reader._right_wrist_camera = None
    reader._head_camera_choices = None
    reader._multiview = True
    reader._camera_layout = ['head', 'wrist', None]
    reader._height, reader._width = 384, 320
    reader._num_video_frames = 5
    colors = {'head': 40, 'wrist': 80, 'head_right': 120, 'wrist_right': 160}
    def decode(camera, *args, kind):
        return [Image.fromarray(np.full((128,128,3),colors[camera],np.uint8))]
    monkeypatch.setattr(reader, '_decode_one_camera', decode)
    left = reader._decode_window_video(None,0,0,np.array([0]),0)
    right = reader._decode_window_video(None,0,0,np.array([0]),0,camera_suffix='_right')
    assert len(left) == len(right) == 5
    assert np.asarray(left[0])[0,0,0] == 40
    assert np.asarray(right[0])[0,0,0] == 120
    assert np.asarray(left[0])[300,30,0] == 80
    assert np.asarray(right[0])[300,30,0] == 160
    assert np.asarray(right[0])[300,250,0] == 0
    del colors['wrist_right']
    with pytest.raises(KeyError):
        reader._decode_window_video(None,0,0,np.array([0]),0,camera_suffix='_right')


def test_canonical_config_and_registry_use_native_action_reader() -> None:
    config = yaml.safe_load(Path("configs/dataloader/libero.yaml").read_text(encoding="utf-8"))
    assert config["type"] == "libero"
    assert config["action_mode"] == "eef"
    assert DATASET_REGISTRY["libero"] is LiberoDataset


def test_reader_hard_preserves_rot6d_under_custom_stats() -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    # Deliberately non-identity rotation stats: the reader must still leave
    # rot6d untouched rather than relying only on the generated stats artifact.
    reader._normalization_stats = {
        "min": np.zeros(10, dtype=np.float32),
        "max": np.ones(10, dtype=np.float32),
    }
    raw = np.array(
        [[0.25, 0.5, 0.75, 0.998, 0.02, -0.04, 0.01, 0.999, 0.02, -0.5]],
        dtype=np.float32,
    )
    normalized = reader._normalize_array(raw)
    np.testing.assert_array_equal(normalized[:, 3:9], raw[:, 3:9])
    expected_non_rot = np.clip(raw[:, [0, 1, 2, 9]] * 2.0 - 1.0, -1.0, 1.0)
    np.testing.assert_allclose(normalized[:, [0, 1, 2, 9]], expected_non_rot)


def test_reader_rejects_incomplete_compatibility_stats(tmp_path) -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    reader._source_stats_path = str(tmp_path / "normalization_stats.npy")
    reader._dataset_dir = tmp_path
    reader._raw_action_dim = 10
    np.save(reader._source_stats_path, {"eef": {}}, allow_pickle=True)
    with pytest.raises(KeyError, match="eef"):
        reader._load_stats({})


def test_multiview_target_camera_does_not_disable_wrist_stream() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = True
    reader._target_camera = "observation.images.image"
    reader._camera_layout_param = [
        "observation.images.image",
        "observation.images.image2",
        None,
    ]
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.image": {"dtype": "video"},
        "observation.images.image2": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.image",
        "observation.images.image2",
        None,
    )


def test_libero_camera_resolution_falls_back_to_info_features() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = True
    reader._target_camera = "observation.images.image"
    reader._camera_layout_param = None
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.agentview_image": {"dtype": "video"},
        "observation.images.wrist_image": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.agentview_image",
        "observation.images.wrist_image",
        None,
    )


def test_single_view_target_camera_still_wins() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = False
    reader._target_camera = "observation.images.image2"
    reader._camera_layout_param = None
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    assert reader._resolve_cameras({"features": {}}) == (
        "observation.images.image2",
        None,
        None,
    )


def test_explicit_layout_falls_back_to_legacy_wrist_feature() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = True
    reader._target_camera = "observation.images.image"
    reader._camera_layout_param = [
        "observation.images.image",
        "observation.images.image2",
        None,
    ]
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.image": {"dtype": "video"},
        "observation.images.wrist_image": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.image",
        "observation.images.wrist_image",
        None,
    )
    assert reader._camera_layout_param == [
        "observation.images.image",
        "observation.images.wrist_image",
        None,
    ]


def test_single_view_explicit_layout_is_honored_without_target() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = False
    reader._target_camera = None
    reader._camera_layout_param = ["observation.images.agentview_image", None, None]
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.image": {"dtype": "video"},
        "observation.images.agentview_image": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.agentview_image",
        None,
        None,
    )
