import torch

from torchdyno.models.esn.reservoir import Reservoir


def _const_hook(value):
    def hook(input, initial_state, mask=None):
        for _ in range(input.shape[0]):
            yield torch.full((input.shape[1], 4), value)

    return hook


def test_default_forward_uses_no_hook():
    torch.manual_seed(0)
    res = Reservoir(input_size=2, hidden_size=4)
    out = res(torch.randn(5, 3, 2))
    assert out.shape == (5, 3, 4)


def test_register_state_hook_overrides_forward():
    res = Reservoir(input_size=2, hidden_size=4)
    res.register_state_hook(_const_hook(0.5))
    out = res(torch.randn(5, 3, 2))
    assert out.shape == (5, 3, 4)
    assert torch.allclose(out, torch.full((5, 3, 4), 0.5))


def test_clear_state_hook_restores_default():
    torch.manual_seed(0)
    res = Reservoir(input_size=2, hidden_size=4)
    x = torch.randn(5, 3, 2)
    before = res(x)
    res.register_state_hook(_const_hook(0.5))
    res.clear_state_hook()
    after = res(x)
    assert torch.allclose(before, after)
