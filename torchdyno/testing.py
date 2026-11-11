"""Checks for layers that run a cell over time.

``assert_layer_conforms`` checks a layer such as
:class:`~torchdyno.sequence.Recurrent` against the plain loop over its cell. Use it
in the tests of a new cell or layer. It raises ``AssertionError`` and needs no
test framework. ``echo_state_distance`` measures how fast a layer forgets where
it started.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.testing import assert_close

from torchdyno.learners.scoring import restore_modes

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.sequence import Recurrent

__all__ = [
    "assert_layer_conforms",
    "echo_state_distance",
]


def assert_layer_conforms(layer: Recurrent, x: Tensor) -> Tensor:
    """Assert that ``layer`` runs its ``cell`` over ``x`` like the plain loop.

    Checks the shape and dtype, the match with a step-by-step loop over
    ``layer.cell``, each sequence of the batch running independently,
    continuing from ``h0``, gradients reaching the input, and every parameter
    that requires grad receiving a gradient. The layer runs in eval mode, so
    dropout does not make the runs differ, and is left in the modes it had.

    Args:
        layer: a :class:`~torchdyno.sequence.Recurrent` layer or a subclass.
        x: a ``(T, B, F)`` input with at least two timesteps and two sequences.

    Returns:
        ``layer(x)``, for further assertions.

    Raises:
        ValueError: if ``x`` is too small to check with.
        AssertionError: if a check fails.
    """
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 2:  # noqa: PLR2004
        raise ValueError(
            f"Need a (T, B, F) input with T, B >= 2 to check, got {tuple(x.shape)}."
        )
    with restore_modes(layer):
        return _check(layer.eval(), x)


def _check(layer: Recurrent, x: Tensor) -> Tensor:
    cell = layer.cell
    out = layer(x)
    expected = (x.shape[0], x.shape[1], cell.state_size)
    assert out.shape == expected, f"output shape {tuple(out.shape)} != {expected}"
    assert out.dtype == x.dtype, f"output dtype {out.dtype} != input dtype {x.dtype}"

    h = x.new_zeros(x.shape[1], cell.state_size)
    loop = []
    for u in x:
        h = cell(u, h)
        loop.append(h)
    assert_close(out, torch.stack(loop), msg="output must match the loop over the cell")
    assert_close(
        layer(x[:, :1]),
        out[:, :1],
        msg="each sequence must run independently of the rest of the batch",
    )

    half = x.shape[0] // 2
    assert_close(
        layer(x[half:], h0=out[half - 1]),
        out[half:],
        msg="starting from h0 must continue the sequence",
    )

    trainable = [(n, p) for n, p in layer.named_parameters() if p.requires_grad]
    x_req = x.detach().clone().requires_grad_(True)
    states = layer(x_req)
    assert states.requires_grad, "the output must stay attached to the input"
    grads = torch.autograd.grad(
        states.sum(), [x_req, *(p for _, p in trainable)], allow_unused=True
    )
    assert grads[0] is not None and bool(grads[0].ne(0).any()), (
        "gradients must reach the input"
    )
    unused = [n for (n, _), g in zip(trainable, grads[1:], strict=True) if g is None]
    assert not unused, f"trainable parameters receive no gradient: {unused}"
    return out


@torch.no_grad()
def echo_state_distance(layer: Recurrent, x: Tensor, scale: float = 1.0) -> Tensor:
    """How far apart two runs of ``layer`` on ``x`` stay, from random starts.

    With the echo state property the input washes the starting state out, so
    the distance decays to zero; if it does not, the layer's output depends on
    where it started, not only on its input. The starts are drawn from torch's
    generator, and the layer runs in eval mode.

    Args:
        layer: a :class:`~torchdyno.sequence.Recurrent` layer or a subclass.
        x: a ``(T, B, F)`` input.
        scale: the standard deviation of the random starting states.

    Returns:
        The ``(T,)`` distance between the two runs' states at every timestep,
        averaged over the batch.
    """
    shape = (x.shape[1], layer.cell.state_size)
    starts = [scale * torch.randn(shape, dtype=x.dtype, device=x.device) for _ in "ab"]
    with restore_modes(layer):
        a, b = (layer.eval()(x, h0=start) for start in starts)
    return (a - b).norm(dim=-1).mean(dim=1)
