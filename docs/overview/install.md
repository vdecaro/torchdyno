# Install TorchDyno

TorchDyno can be installed from [PyPI](https://pypi.org/project/torchdyno/).
It requires Python `>=3.11,<3.15` and [PyTorch](https://pytorch.org/),
which is installed with it. To choose a CPU or CUDA build of PyTorch, install
it first as its documentation describes.

!!! note

    Do not use the global environment to install TorchDyno.
    It is recommended to create a [virtual environment](https://docs.python.org/3/library/venv.html) first.

## Using pip

Create and activate a virtual environment, then install the package:

```shell
python -m venv .venv
source .venv/bin/activate
pip install torchdyno
```

## Using uv

If your project uses [uv](https://docs.astral.sh/uv/), add TorchDyno to the
project dependencies:

```shell
uv add torchdyno
```

## Installing from source

For development, clone the repository and install the dependency groups you
need:

```shell
git clone https://github.com/vdecaro/torchdyno.git
cd torchdyno
uv sync --group dev --group test
```

The documentation dependencies live in the `docs` group:

```shell
uv sync --group docs
```

Now you are ready to build [your first TorchDyno model](getting-started/index.md)!
