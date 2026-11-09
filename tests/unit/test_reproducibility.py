import random

import torch

from torchdyno import (
    get_rng_state,
    seed_all,
    set_rng_state,
)


def test_seed_all_repeats_python_and_torch_draws():
    seed_all(123, deterministic=False)
    first = random.random(), torch.randn(5)
    seed_all(123, deterministic=False)
    second = random.random(), torch.randn(5)
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_seed_all_enables_deterministic_algorithms():
    was = torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark
    try:
        seed_all(0)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(was[0])
        torch.backends.cudnn.benchmark = was[1]


def test_rng_state_round_trip():
    state = get_rng_state()
    first = random.random(), torch.randn(5)
    set_rng_state(state)
    assert first[0] == random.random()
    assert torch.equal(first[1], torch.randn(5))
