import torch

from torchdyno.heads import RegressionHead
from torchdyno.model import SequenceModel
from torchdyno.models.esn.core import ESNCore
from torchdyno.optim.ridge_regression import fit_readout


def test_esn_core_readout_fit_learns_reconstruction():
    torch.manual_seed(0)
    core = ESNCore(input_size=1, layer_sizes=[100], input_scaling=0.9, rho=0.9)
    head = RegressionHead(core.state_size, 1)
    model = SequenceModel(core, head)

    # Task: reconstruct the current input from the reservoir state.
    T, B = 300, 4
    x = torch.randn(T, B, 1)
    y = x.clone()

    # A "loader" is any iterable of (x, y) batches; one batch suffices here.
    loader = [(x, y)]

    readout, _a, _b = fit_readout(
        loader,
        preprocess_fn=lambda batch: core(batch).states,
        l2=1e-6,
    )
    # fit_readout returns (output_size, hidden_size); the head weight is
    # (hidden_size, output_size).
    head.set_weight(readout.T)

    pred = model(x)
    assert pred.shape == (T, B, 1)

    mse = torch.mean((pred - y) ** 2).item()
    var = torch.var(y).item()
    # The fitted readout must substantially beat a mean/zero predictor.
    assert mse < 0.5 * var
