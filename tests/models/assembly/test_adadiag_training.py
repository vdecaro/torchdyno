import torch

from torchdyno.heads import ClassificationHead
from torchdyno.model import SequenceModel
from torchdyno.models.assembly.adadiag import AdaDiagCore


def _build():
    core = AdaDiagCore(input_size=1, block_sizes=[16, 16], coupling_topology=[(0, 1)])
    head = ClassificationHead(core.state_size, num_classes=2, pool="last")
    return SequenceModel(core, head)


def test_gradients_flow_to_all_trainable_parameters():
    torch.manual_seed(0)
    model = _build()
    x = torch.randn(20, 8, 1)
    y = (x.mean(dim=0).squeeze(-1) > 0).long()  # (B,)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    # Only trainable params should receive gradients; frozen buffers held as
    # non-trainable Parameters (e.g. the coupling sparsity mask) legitimately
    # have no grad.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    for name, p in trainable:
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_loss_decreases_under_adam():
    torch.manual_seed(0)
    model = _build()
    # Tiny learnable task: class = sign of the mean of the input sequence.
    x = torch.randn(30, 16, 1)
    y = (x.mean(dim=0).squeeze(-1) > 0).long()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossfn = torch.nn.CrossEntropyLoss()

    first = None
    last = None
    for i in range(60):
        opt.zero_grad()
        loss = lossfn(model(x), y)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()

    assert last < first  # training reduces the loss (gradients are useful)
