import torch

import torchdyno
from torchdyno.heads import ClassificationHead, RegressionHead
from torchdyno.model import SequenceModel
from torchdyno.models.base import CoreOutput
from torchdyno.models.esn.core import ESNCore


def test_top_level_exports():
    assert hasattr(torchdyno, "SequenceModel")
    assert hasattr(torchdyno, "heads")


def test_regression_model_forward_shape():
    torch.manual_seed(0)
    core = ESNCore(input_size=1, layer_sizes=[16])
    model = SequenceModel(core, RegressionHead(core.state_size, 1))
    out = model(torch.randn(12, 3, 1))
    assert out.shape == (12, 3, 1)


def test_classification_model_forward_shape():
    torch.manual_seed(0)
    core = ESNCore(input_size=1, layer_sizes=[16, 8], arch_type="multi")
    model = SequenceModel(core, ClassificationHead(core.state_size, 5, pool="mean"))
    out = model(torch.randn(12, 3, 1))
    assert out.shape == (3, 5)


def test_return_core_output():
    torch.manual_seed(0)
    core = ESNCore(input_size=1, layer_sizes=[16])
    model = SequenceModel(core, RegressionHead(core.state_size, 2))
    pred, core_out = model(torch.randn(12, 3, 1), return_core_output=True)
    assert pred.shape == (12, 3, 2)
    assert isinstance(core_out, CoreOutput)
