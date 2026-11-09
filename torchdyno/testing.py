"""Conformance checks for layers that run a cell over time.

``assert_layer_conforms`` checks a layer such as
:class:`~torchdyno.layers.Recurrent` against the plain loop over its cell. Use it
in the tests of a new cell or layer. It raises ``AssertionError`` and needs no
test framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.testing import assert_close

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.layers import Recurrent


def assert_layer_conforms(layer: Recurrent, x: Tensor) -> Tensor:
    """Assert that ``layer`` runs its ``cell`` over ``x`` like the plain loop.

    Checks the shape and dtype, the match with a step-by-step loop over
    ``layer.cell``, continuing from ``h0``, gradients reaching the input, and
    every parameter that requires grad receiving a gradient.

    Args:
        layer: a :class:`~torchdyno.layers.Recurrent` layer or a subclass.
        x: a ``(T, B, F)`` input, at least two timesteps long.

    Returns:
        ``layer(x)``, for further assertions.

    Raises:
        AssertionError: if a check fails.
    """
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
