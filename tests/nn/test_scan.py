import torch

from torchdyno.nn.scan import associative_scan


def _loop_reference(a, b):
    h = torch.zeros_like(b[0])
    out = []
    for t in range(b.shape[0]):
        h = a[t] * h + b[t]
        out.append(h)
    return torch.stack(out, dim=0)


def test_scan_matches_loop_real():
    torch.manual_seed(0)
    a = torch.rand(23, 4, 5)          # arbitrary length (not a power of two)
    b = torch.randn(23, 4, 5)
    assert torch.allclose(associative_scan(a, b), _loop_reference(a, b), atol=1e-5)


def test_scan_matches_loop_complex():
    torch.manual_seed(0)
    a = torch.polar(torch.rand(16, 3, 6), torch.rand(16, 3, 6))   # |a|<1
    b = torch.randn(16, 3, 6, dtype=torch.complex64)
    assert torch.allclose(associative_scan(a, b), _loop_reference(a, b), atol=1e-5)


def test_scan_length_one():
    a = torch.rand(1, 2, 2)
    b = torch.randn(1, 2, 2)
    assert torch.allclose(associative_scan(a, b), b, atol=1e-6)


def test_scan_is_differentiable():
    torch.manual_seed(0)
    a = torch.rand(8, 2, 3, dtype=torch.double, requires_grad=True)
    b = torch.randn(8, 2, 3, dtype=torch.double, requires_grad=True)
    torch.autograd.gradcheck(associative_scan, (a, b))
