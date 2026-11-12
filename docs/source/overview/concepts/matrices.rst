========
Matrices
========

The lowest level decides how a weight matrix is structured, initialised and
applied. Every matrix is an ``nn.Module`` with ``in_features`` and
``out_features`` whose ``forward(x)`` applies ``x Wᵀ``, without building ``W``
when its structure allows it, and whose ``dense()`` returns ``W``. Cells take
their matrices as arguments, so the same update rule works with any structure,
and ``torch.nn.Linear`` works wherever a matrix is expected
(:class:`~torchdyno.matrices.LinearMap` names that contract).

A matrix's ``weight`` is its storage, which can differ from ``W``: masked,
bounded or, for a complex matrix, split into real and imaginary parts. Read
``W`` through ``dense()`` or :func:`~torchdyno.matrices.to_dense`, never
through ``weight``.

Structures
----------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Matrix
     - Structure
     - Key arguments
   * - :class:`~torchdyno.matrices.Dense`
     - a full matrix with an optional bias; real or complex
     - ``in_features``, ``out_features``, ``init``, ``bias``, ``dtype``
   * - :class:`~torchdyno.matrices.Diagonal`
     - a diagonal, optionally through a bound such as ``torch.tanh``
     - ``size``, ``init``, ``bound``
   * - :class:`~torchdyno.matrices.BlockDiagonal`
     - square blocks on the diagonal, orthogonal by default, optionally bounded
     - ``sizes``, ``init``, ``bound``
   * - :class:`~torchdyno.matrices.SkewCoupling`
     - skew-symmetric couplings between chosen pairs of blocks
     - ``sizes``, ``pairs``, ``init``

:class:`~torchdyno.matrices.Dense` is the default choice and the only one with
a bias. Without ``init`` it draws from ``U(±1/sqrt(in_features))``, as
``nn.Linear`` does; without ``bias`` it has none, so give ``bias=`` an
initialiser where a cell's units need one.

The other three are square and exist for structure that has to survive
training. A :class:`~torchdyno.matrices.Diagonal` gives every unit a
self-weight and no coupling. A :class:`~torchdyno.matrices.BlockDiagonal`
gives independent modules, and its zero entries stay zero: the mask is the
nonzero pattern of the initial blocks, so blocks drawn with a sparse
initialiser stay sparse. A :class:`~torchdyno.matrices.SkewCoupling` stores only
the coupled pairs of blocks and returns ``C - Cᵀ``, so the coupling remains
skew-symmetric however its parameters change.

.. code-block:: python

   import torch
   from torch import nn

   from torchdyno.matrices import (
       BlockDiagonal,
       Dense,
       Diagonal,
       SkewCoupling,
       to_dense,
   )

   modules = BlockDiagonal([10] * 10)  # ten orthogonal 10 x 10 blocks
   coupling = SkewCoupling([10] * 10, pairs="ring")  # block i with block i + 1
   encoder = Dense(3, 100, bias=nn.init.zeros_)
   x = torch.randn(8, 100)
   print(modules(x).shape, encoder(torch.randn(8, 3)).shape)
   torch.testing.assert_close(modules(x), x @ to_dense(modules).T)
   torch.testing.assert_close(to_dense(coupling), -to_dense(coupling).T)

``pairs`` of a skew coupling is a number of random pairs, a float fraction of
all pairs, ``"ring"`` or an explicit list of ``(i, j)`` pairs.

Initialisers
------------

An initialiser is a callable that fills a tensor in place, as in
``torch.nn.init``, so every ``torch.nn.init`` function works as one.
TorchDyno adds four that build on it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Initialiser
     - Draws
   * - :func:`~torchdyno.matrices.uniform`
     - ``U(-scale, scale)``; ``scale`` may be a tensor that broadcasts to the
       matrix, e.g. one per input column
   * - :func:`~torchdyno.matrices.normal`
     - ``N(0, std²)``, with the same option of a tensor of scales
   * - :func:`~torchdyno.matrices.cycle`
     - a ring: unit ``i`` feeds unit ``i + 1``, every connection of one weight
   * - :func:`~torchdyno.matrices.spectral`
     - the draw of another initialiser, rescaled to a spectral radius

:func:`~torchdyno.matrices.spectral` wraps another initialiser, which is how a
reservoir gets its spectral radius whatever its sparsity or distribution.
Complex matrices draw their real and imaginary parts independently.

.. code-block:: python

   from torchdyno.matrices import cycle, normal, spectral, spectral_radius, uniform

   scaled = Dense(3, 100, init=uniform(torch.tensor([1.0, 0.5, 0.1])))
   reservoir = Dense(100, 100, init=spectral(normal(1.0), 0.9))
   sparse = Dense(100, 100, init=spectral(lambda w: nn.init.sparse_(w, 0.9), 0.9))
   ring = Dense(100, 100, init=cycle(0.9))
   print(spectral_radius(reservoir), spectral_radius(ring))  # 0.9 0.9

.. note::

   ``spectral`` raises ``ValueError`` when the draw has only zero eigenvalues,
   such as a delay line, because no scale gives it a spectral radius.
   ``spectral_radius`` returns the largest absolute eigenvalue of any square
   matrix, ``nn.Linear`` included, to check what an initialiser produced.

Your own initialiser is any function of one tensor. Fill it in place and return
it:

.. code-block:: python

   def orthogonal_scaled(tensor):
       return nn.init.orthogonal_(tensor, gain=0.9)

   assert abs(spectral_radius(Dense(50, 50, init=orthogonal_scaled)) - 0.9) < 1e-4

Bounds and freezing
-------------------

A bound maps the stored entries to the matrix on every use. ``Diagonal(100,
bound=torch.tanh)`` keeps every self-weight in ``(-1, 1)`` however far training
pushes the stored entries. A :class:`~torchdyno.matrices.BlockDiagonal` bound
must map zero to zero, or the masked entries would leak, and raises
``ValueError`` otherwise.

Freeze any matrix with ``requires_grad_(False)``. Freezing only stops gradients:
it never changes the initialisation, so a frozen reservoir is the random
reservoir it was drawn as, and a frozen matrix can still be updated by a local
rule (see below).

.. code-block:: python

   stable = Diagonal(100, bound=torch.tanh)
   frozen = BlockDiagonal([10] * 10, bound=torch.tanh).requires_grad_(False)
   print(sum(p.requires_grad for p in stable.parameters()),
         sum(p.requires_grad for p in frozen.parameters()))  # 1 0

Complex matrices
----------------

``Dense(3, 64, dtype=torch.cfloat)`` stores its real and imaginary parts in a
last dimension of 2, so casts such as ``.double()`` reach both parts, and
``dense()`` returns the complex matrix. Real inputs are cast on the way in.
Diagonal and block structures take a complex ``dtype`` as well.

.. code-block:: python

   complex_dense = Dense(3, 64, dtype=torch.cfloat)
   print(complex_dense.weight.shape, complex_dense.dense().shape)  # (64, 3, 2) (64, 3)
   print(complex_dense(torch.randn(5, 3)).dtype)  # torch.complex64

.. warning::

   ``weight`` of a complex matrix is real storage with a trailing dimension of
   2. Do not read it as the matrix: use ``dense()``.

Reading and updating any matrix
-------------------------------

Three functions work on every matrix, an ``nn.Linear`` included.
:func:`~torchdyno.matrices.to_dense` returns ``W``;
:func:`~torchdyno.matrices.spectral_radius` returns its largest absolute
eigenvalue in double precision; and :func:`~torchdyno.matrices.update_` adds an
update of the dense ``W`` as far as the structure allows. The last is what
local learning rules need: a diagonal takes the diagonal of the update, a
block-diagonal matrix its blocks and a skew coupling its skew part, so the
structure survives. Frozen parameters are updated too.

.. code-block:: python

   from torchdyno.matrices import update_

   diagonal = Diagonal(100)
   before = to_dense(diagonal).clone()
   update_(diagonal, torch.ones(100, 100))
   torch.testing.assert_close(to_dense(diagonal), before + torch.eye(100))

Choosing a structure
--------------------

- A reservoir whose recurrent weights are not trained: ``Dense`` with
  ``spectral(...)``, or ``cycle`` for the cheapest ring. See
  :doc:`../examples/esn`.
- Per-unit dynamics with learnt stable self-weights: ``Diagonal`` with a
  ``tanh`` bound.
- Independent modules coupled sparsely: ``BlockDiagonal`` plus ``SkewCoupling``,
  as in :doc:`../examples/scn` and :doc:`../examples/adadiag`.
- Complex modes: ``Dense`` with ``dtype=torch.cfloat``, or the diagonal
  transitions of :doc:`../examples/lru`.

For a structure of your own, such as a low-rank matrix, write a module with
``in_features``, ``out_features``, ``forward`` and ``dense()``; the rest of the
library then accepts it. :doc:`../advanced/custom-parts` shows how.
