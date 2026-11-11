import pytest
import torch
from torch import nn

from tests.unit.compositions import (
    lru,
    reservoir,
)
from torchdyno.learners import (
    Ridge,
    evaluate,
)
from torchdyno.readout import (
    PPV,
    Last,
    Max,
    Mean,
)
from torchdyno.sequence import (
    Concat,
    RandomConv,
    Scan,
)


def _batch(short, long, where):
    """``short`` padded with NaN to ``long``'s length, at its end or start."""
    pad = torch.full((len(long) - len(short), *short.shape[1:]), torch.nan)
    padded = torch.cat([short, pad] if where == "end" else [pad, short])
    return torch.cat([padded, long], dim=1)


LAYERS = {
    "recurrent": lambda: reservoir(features=2),
    "scan": lambda: Scan(lru(features=2).cell),
}


@pytest.mark.parametrize("where", ["end", "start"])
@pytest.mark.parametrize("make", LAYERS.values(), ids=LAYERS.keys())
def test_a_padded_sequence_runs_as_it_would_alone(make, where):
    torch.manual_seed(0)
    layer, short, long = make(), torch.randn(5, 1, 2), torch.randn(8, 1, 2)
    out = layer(_batch(short, long, where))
    alone = layer(short)[:, 0]
    produced = out[:5, 0] if where == "end" else out[3:, 0]
    torch.testing.assert_close(produced, alone)
    assert (out[5:, 0] if where == "end" else out[:3, 0]).isnan().all()
    torch.testing.assert_close(out[:, 1], layer(long)[:, 0])
    for pool in (Last(), Mean()):
        torch.testing.assert_close(pool(out)[0], pool(alone[:, None])[0])


@pytest.mark.parametrize("where", ["end", "start"])
def test_random_kernels_see_a_padded_series_as_they_would_alone(where):
    torch.manual_seed(0)
    layer = RandomConv(1, length=20, kernels=200)
    features = Concat(PPV(), Max())
    short, long = torch.randn(20, 1, 1), torch.randn(30, 1, 1)
    batched = features(layer(_batch(short, long, where)))
    torch.testing.assert_close(batched[0], features(layer(short))[0])
    assert layer.paddings.gt(0).any() and layer.paddings.eq(0).any()


def test_ridge_and_evaluate_leave_out_the_steps_a_sequence_does_not_have():
    torch.manual_seed(0)
    short, long = torch.randn(5, 1, 3), torch.randn(8, 1, 3)
    y_short, y_long = torch.randn(5, 1, 2), torch.randn(8, 1, 2)
    model = nn.Sequential(nn.Identity(), nn.Linear(3, 2))
    Ridge(1e-3).fit(
        model, [(_batch(short, long, "end"), _batch(y_short, y_long, "end"))]
    )
    reference = nn.Sequential(nn.Identity(), nn.Linear(3, 2))
    Ridge(1e-3).fit(reference, [(short, y_short), (long, y_long)])
    torch.testing.assert_close(model[-1].weight, reference[-1].weight)
    seen = {}

    def probe(pred, target):
        seen["rows"] = (len(pred), len(target))
        return 0.0

    evaluate(
        model, [(_batch(short, long, "end"), _batch(y_short, y_long, "end"))], probe
    )
    assert seen["rows"] == (5 + 8, 5 + 8)
