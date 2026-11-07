from importlib.metadata import version

import torchdyno


def test_version_matches_the_installed_package():
    assert torchdyno.__version__ == version("torchdyno")
