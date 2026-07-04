"""Coupled-modular RNN assemblies (AdaDiag, SCN) as SequenceCores."""

from .block_diagonal import BlockDiagonal
from .skew_symm_coupling import SkewAntisymmetricCoupling
from ._common import build_coupling
from .adadiag import AdaDiagCore, FrequencyGate
from .scn import SCNCore

__all__ = [
    "BlockDiagonal",
    "SkewAntisymmetricCoupling",
    "build_coupling",
    "FrequencyGate",
    "AdaDiagCore",
    "SCNCore",
]
