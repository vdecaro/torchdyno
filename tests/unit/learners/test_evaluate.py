import torch
from torch import nn

from tests.unit.compositions import reservoir
from torchdyno.learners import evaluate
from torchdyno.pooling import Last


def _concat_check(pooled, y_shape, cat_dim):
    torch.manual_seed(0)
    parts = (
        [reservoir(1, 8), Last(), nn.Linear(8, 2)]
        if pooled
        else [reservoir(1, 8), nn.Linear(8, 2)]
    )
    batches = [(torch.randn(5, b, 1), torch.randn(*y_shape(b))) for b in (3, 2)]
    seen = {}

    def probe(pred, target):
        seen["pred"], seen["target"] = pred, target
        return 0.0

    evaluate(nn.Sequential(*parts), batches, probe)
    assert torch.equal(seen["target"], torch.cat([y for _, y in batches], dim=cat_dim))
    assert seen["pred"].shape[cat_dim] == 5


def test_concatenates_time_first_outputs_on_the_batch_dim():
    _concat_check(False, lambda b: (5, b, 2), cat_dim=1)


def test_concatenates_pooled_outputs_on_dim_zero():
    _concat_check(True, lambda b: (b, 2), cat_dim=0)
