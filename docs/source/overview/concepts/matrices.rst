========
Matrices
========

Every matrix is an ``nn.Module`` with ``in_features`` and ``out_features``
whose ``forward(x)`` applies ``x Wᵀ``, without building ``W`` when its
structure allows it, and whose ``dense()`` returns ``W``. Cells take their
matrices as arguments, so the same update rule works with any structure, and
``torch.nn.Linear`` works wherever a matrix is expected.

Structures
----------

.. list-table::
   :header-rows: 1

   * - Matrix
     - Structure
   * - :class:`~torchdyno.matrices.Dense`
     - a full matrix with an optional bias; real or complex
   * - :class:`~torchdyno.matrices.Diagonal`
     - a diagonal, optionally through a bound such as ``torch.tanh``
   * - :class:`~torchdyno.matrices.BlockDiagonal`
     - square blocks on the diagonal, orthogonal by default, optionally bounded
   * - :class:`~torchdyno.matrices.SkewCoupling`
     - skew-symmetric couplings between chosen pairs of blocks

Initialisers
------------

An initialiser is a callable that fills a tensor in place, as in
``torch.nn.init``, so every ``torch.nn.init`` function works as one.
:func:`~torchdyno.matrices.uniform` and :func:`~torchdyno.matrices.normal` take
a scale, or a tensor of scales that broadcasts to the matrix, e.g. one per input
channel; :func:`~torchdyno.matrices.spectral` rescales another initialiser's
draw to a spectral radius; :func:`~torchdyno.matrices.cycle` builds a ring.

.. code-block:: python

   import torch
   from torch import nn

   from torchdyno.matrices import (
       BlockDiagonal,
       Dense,
       Diagonal,
       SkewCoupling,
       cycle,
       normal,
       spectral,
       spectral_radius,
       to_dense,
       uniform,
       update_,
   )

   scaled = Dense(3, 100, init=uniform(torch.tensor([1.0, 0.5, 0.1])))
   reservoir = Dense(100, 100, init=spectral(normal(1.0), 0.9))
   sparse = Dense(100, 100, init=lambda w: nn.init.sparse_(w, 0.9))
   ring = Dense(100, 100, init=cycle(0.9))
   print(spectral_radius(reservoir), spectral_radius(ring))  # 0.9 0.9

Bounds and freezing
-------------------

A bound maps the stored entries to the matrix, e.g. ``Diagonal(100,
bound=torch.tanh)`` keeps every self-weight in ``(-1, 1)`` while training.
Freeze any matrix with ``requires_grad_(False)``; freezing never changes its
initialisation.

.. code-block:: python

   modules = BlockDiagonal([10] * 10, bound=torch.tanh).requires_grad_(False)
   coupling = SkewCoupling([10] * 10, pairs=20)  # 20 random pairs of blocks
   x = torch.randn(8, 100)
   torch.testing.assert_close(modules(x), x @ to_dense(modules).T)

Complex matrices
----------------

``Dense(3, 64, dtype=torch.cfloat)`` stores its real and imaginary parts in a
last dimension of 2, so casts such as ``.double()`` reach both parts, and
``dense()`` returns the complex matrix. Real inputs are cast on the way in.

Reading and updating any matrix
-------------------------------

:func:`~torchdyno.matrices.to_dense` returns ``W`` for any matrix, an
``nn.Linear`` included. :func:`~torchdyno.matrices.update_` adds an update of
the dense ``W`` as far as the structure allows, which is what local learning
rules need: a diagonal takes the diagonal of the update, a block-diagonal
matrix its blocks and a skew coupling its skew part.

.. code-block:: python

   diagonal = Diagonal(100)
   before = to_dense(diagonal).clone()
   update_(diagonal, torch.ones(100, 100))
   torch.testing.assert_close(to_dense(diagonal), before + torch.eye(100))
