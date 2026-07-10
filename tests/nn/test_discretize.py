import torch

from torchdyno.nn.discretize import zoh


def test_a_bar_is_stable_for_negative_real_A():
    torch.manual_seed(0)
    A = torch.complex(-torch.rand(50) - 0.1, torch.randn(50))  # Re < 0
    dt = torch.rand(50) * 0.1 + 1e-3  # dt > 0
    B = torch.randn(50, 3, dtype=torch.complex64)
    a_bar, b_bar = zoh(A, dt, B)
    assert a_bar.shape == (50,) and b_bar.shape == (50, 3)
    assert (a_bar.abs() < 1.0).all()


def test_zoh_matches_augmented_matrix_exp():
    # ZOH via the closed form must equal expm([[A, B], [0, 0]]·dt) per mode
    # (an independent code path from the (exp-1)/A division).
    torch.manual_seed(0)
    A = torch.complex(-torch.rand(8) - 0.5, torch.randn(8))
    dt = torch.rand(8) * 0.05 + 1e-2
    B = torch.randn(8, 1, dtype=torch.complex64)
    a_bar, b_bar = zoh(A, dt, B)
    for n in range(8):
        m = torch.zeros(2, 2, dtype=torch.complex64)
        m[0, 0] = A[n] * dt[n]
        m[0, 1] = B[n, 0] * dt[n]
        em = torch.matrix_exp(m)
        assert torch.allclose(a_bar[n], em[0, 0], atol=1e-5)
        assert torch.allclose(b_bar[n, 0], em[0, 1], atol=1e-5)


def test_zoh_small_dt_euler_limit():
    A = torch.complex(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, -0.5]))
    dt = torch.full((2,), 1e-6)
    B = torch.ones(2, 1, dtype=torch.complex64)
    a_bar, b_bar = zoh(A, dt, B)
    assert torch.allclose(a_bar, 1.0 + dt * A, atol=1e-8)
    assert torch.allclose(b_bar, dt.unsqueeze(-1) * B, atol=1e-8)
