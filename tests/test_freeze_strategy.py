"""Test that the freeze strategy correctly freezes/unfreezes model components."""

import torch.nn as nn


def test_no_grad_wrapper_uses_copied_subtree():
    import copy
    import torch
    from openwam.model.architectures.base import _wrap_forward_in_no_grad

    source = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        source[0].weight.fill_(1)
    _wrap_forward_in_no_grad(source)
    copied = copy.deepcopy(source)
    with torch.no_grad():
        copied[0].weight.fill_(3)

    inputs = torch.ones(1, 1, requires_grad=True)
    result = copied(inputs)
    assert result.item() == 3
    assert source(inputs).item() == 1
    assert not result.requires_grad
    # Direct entry into a copied descendant must still suppress gradients.
    assert not copied[0](inputs).requires_grad
    forward = copied.forward
    _wrap_forward_in_no_grad(copied)
    assert copied.forward is forward
    assert list(copied.state_dict()) == ["0.weight"]


class MockPipeline:
    """Minimal mock of WanVideoPipeline with named submodules."""

    def __init__(self):
        self.dit = nn.Linear(10, 10)
        self.vae = nn.Linear(10, 10)
        self.text_encoder = nn.Linear(10, 10)
        self.image_encoder = nn.Linear(10, 10)
        self.vace = nn.Linear(10, 10)


def _apply_freeze(pipe, freeze_list):
    """Replicate the freeze logic from OpenWAMTrainer."""
    for name in freeze_list:
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False)


def test_default_freeze_strategy():
    """Default freeze: text_encoder, vae, image_encoder frozen; dit, vace trainable."""
    pipe = MockPipeline()
    freeze_list = ["text_encoder", "vae", "image_encoder"]
    _apply_freeze(pipe, freeze_list)

    # Frozen
    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in pipe.image_encoder.parameters())

    # Trainable
    assert all(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in pipe.vace.parameters())


def test_freeze_everything_except_dit():
    """FastWAM-style: freeze all except dit."""
    pipe = MockPipeline()
    freeze_list = ["text_encoder", "vae", "image_encoder", "vace"]
    _apply_freeze(pipe, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in pipe.image_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vace.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())


def test_empty_freeze_list():
    """Empty freeze list: everything remains trainable."""
    pipe = MockPipeline()
    _apply_freeze(pipe, [])

    assert all(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in pipe.vae.parameters())
    assert all(p.requires_grad for p in pipe.text_encoder.parameters())
    assert all(p.requires_grad for p in pipe.image_encoder.parameters())
    assert all(p.requires_grad for p in pipe.vace.parameters())


def test_freeze_nonexistent_module():
    """Freezing a module that doesn't exist should not error."""
    pipe = MockPipeline()
    pipe.audio_encoder = None  # explicitly None
    freeze_list = ["text_encoder", "vae", "audio_encoder", "nonexistent_module"]
    _apply_freeze(pipe, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    # dit and others remain trainable
    assert all(p.requires_grad for p in pipe.dit.parameters())
