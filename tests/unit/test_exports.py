import subprocess
import sys

import pytest

import torchdyno

MODULES = [
    "benchmark",
    "learners",
    "matrices",
    "objectives",
    "readout",
    "sequence",
    "testing",
]


def test_a_bare_import_reaches_every_module():
    names = ", ".join(f"torchdyno.{name}" for name in MODULES)
    code = f"import torchdyno; {names}"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", MODULES)
def test_a_star_import_brings_only_the_public_names(name):
    namespace = {}
    exec(f"from torchdyno.{name} import *", namespace)  # noqa: S102
    exported = set(namespace) - {"__builtins__"}
    assert exported == set(getattr(torchdyno, name).__all__)
    assert not exported & {"torch", "nn", "F", "math"}
