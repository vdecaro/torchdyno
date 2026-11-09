"""A small synthetic task shared by the classification examples."""

import math

import torch


def frequency_task(n=512, steps=60, classes=3, noise=0.3, batch=64, seed=0):
    """Noisy sine waves whose class is their frequency.

    Returns train, validation and test batches of time-first ``(x, y)``, with
    ``x`` of shape ``(steps, B, 1)`` and integer labels ``y`` of shape ``(B,)``.
    """
    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 6 * math.pi, steps)
    y = torch.randint(classes, (n,), generator=generator)
    phase = 2 * math.pi * torch.rand(n, generator=generator)
    x = torch.sin((1 + y)[None].float() * t[:, None] + phase[None])
    x = (x + noise * torch.randn(steps, n, generator=generator)).unsqueeze(-1)

    def batches(start, stop):
        return [
            (x[:, i : min(i + batch, stop)], y[i : min(i + batch, stop)])
            for i in range(start, stop, batch)
        ]

    train, val = n * 3 // 5, n * 4 // 5
    return batches(0, train), batches(train, val), batches(val, n)
