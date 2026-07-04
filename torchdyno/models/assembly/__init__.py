"""Coupled-modular RNN assemblies (AdaDiag, SCN) as SequenceCores."""

from ._common import build_coupling
from .adadiag import AdaDiagCore, FrequencyGate
from .scn import SCNCore

__all__ = ["build_coupling", "FrequencyGate", "AdaDiagCore", "SCNCore"]
