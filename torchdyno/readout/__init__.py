"""Level 3, readout: reduce a ``(T, B, H)`` sequence over time to ``(B, H)`` features.

A layer marks positions it did not produce with NaN, as
[`RandomConv`][torchdyno.sequence.RandomConv] does for kernels without padding and every
layer does at the steps a shorter sequence does not have. Every pooling skips
them.

A plain ``nn.Linear`` then maps the features to outputs.
"""

from .pooling import (
    LSPV,
    MIPV,
    MPV,
    PPV,
    Last,
    Max,
    Mean,
)

__all__ = [
    "LSPV",
    "MIPV",
    "MPV",
    "PPV",
    "Last",
    "Max",
    "Mean",
]
