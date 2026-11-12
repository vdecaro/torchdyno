=======
Readout
=======

The readout is the third level: it turns what a sequence layer produced into
the model's outputs. For a label per sequence, a pooling reduces ``(T, B, H)``
to ``(B, H)`` features by reducing over time, and a plain ``nn.Linear`` maps
them to ``(B, O)`` outputs. For a value at every step, such as a forecast, leave
the pooling out: ``nn.Linear`` acts on the last dimension, so ``(T, B, H)``
becomes ``(T, B, O)``.

.. code-block:: python

   import torch
   from torch import nn

   from torchdyno.matrices import Dense, spectral, uniform
   from torchdyno.readout import Last, Mean
   from torchdyno.sequence import LeakyCell, Recurrent

   def reservoir(features, size=64):
       return Recurrent(
           LeakyCell(
               input=Dense(features, size, init=uniform(1.0)),
               recurrent=Dense(size, size, init=spectral(uniform(1.0), 0.9)),
           )
       ).requires_grad_(False)

   x = torch.randn(50, 8, 2)
   per_step = nn.Sequential(reservoir(2), nn.Linear(64, 3))  # a value per step
   per_sequence = nn.Sequential(reservoir(2), Last(), nn.Linear(64, 3))  # a label
   print(per_step(x).shape, per_sequence(x).shape)  # (50, 8, 3) (8, 3)

Pooling
-------

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Pooling
     - Each feature becomes
     - Reach for it when
   * - :class:`~torchdyno.readout.Last`
     - its value at each sequence's last step
     - a recurrent state summarises the sequence
   * - :class:`~torchdyno.readout.Mean`
     - its mean over time
     - the whole sequence matters equally
   * - :class:`~torchdyno.readout.Max`
     - its maximum over time
     - a feature signals that something happened at all
   * - :class:`~torchdyno.readout.PPV`
     - the proportion of positive values (ROCKET)
     - convolution outputs, with a bias that sets the threshold
   * - :class:`~torchdyno.readout.MPV`
     - the mean of the positive values (MultiRocket)
     - how strongly a feature is on, when it is
   * - :class:`~torchdyno.readout.MIPV`
     - the mean index of the positive values (MultiRocket)
     - when a feature tends to be on
   * - :class:`~torchdyno.readout.LSPV`
     - the longest stretch of positive values (MultiRocket)
     - how long a feature stays on

None of them has parameters. The first three suit any feature. The last four
count positive values, so they are meant for outputs centred on zero, such as
those of :class:`~torchdyno.sequence.RandomConv`, whose biases set what counts
as positive.

For a recurrent layer the choice matters: ``Last`` reads a state that has had
to carry everything forward, while ``Mean`` averages the transient in too. Try
both on a task rather than assuming either.

.. code-block:: python

   from torchdyno.readout import LSPV, MIPV, MPV, PPV, Max

   h = torch.randn(40, 8, 16)
   for pooling in (Last(), Mean(), Max(), PPV(), MPV(), MIPV(), LSPV()):
       print(type(pooling).__name__, pooling(h).shape)  # (8, 16) each

Missing steps and NaN
---------------------

A ``NaN`` marks a position a sequence does not have. Every pooling skips them,
which covers both sources: the outputs a convolution without padding did not
produce, and the steps a shorter sequence does not have in a padded batch (see
:doc:`../advanced/variable-length`). ``Last`` returns each sequence's own last
step, not ``x[-1]``; ``Mean`` and ``Max`` ignore the gaps; ``PPV`` divides by
the number of positions that exist.

.. code-block:: python

   states = reservoir(2)(x)
   states[30:, 1] = torch.nan  # the second sequence has 30 steps
   torch.testing.assert_close(Last()(states)[1], states[29, 1])
   torch.testing.assert_close(Mean()(states)[1], states[:30, 1].mean(0))

Two details of the positive-value poolings: ``MPV`` is ``0`` and ``MIPV`` is
``-1`` where a feature is never positive, and ``MIPV`` counts only the
positions that exist, so its indices start at zero at a sequence's first step.

.. warning::

   A NaN that reaches the ``nn.Linear`` is not skipped. The learners drop rows
   whose features are all NaN, but they raise on a row that is NaN in only some
   features, or not finite at all. If you see that error, a layer before the
   pooling produced NaN in a place no pooling covers: pool, or difference after
   the layer rather than before it.

Several poolings at once
------------------------

:class:`~torchdyno.sequence.Concat` runs modules side by side on the same input
and concatenates their outputs, so it pools one sequence several ways. The
linear layer after it is then ``k`` times wider for ``k`` poolings.

.. code-block:: python

   from torchdyno.sequence import Concat, RandomConv

   multirocket = nn.Sequential(
       RandomConv(channels=1, length=60, kernels=500),
       Concat(PPV(), Max(), MPV(), LSPV()),
       nn.Linear(4 * 500, 3),
   )
   print(multirocket(torch.randn(60, 8, 1)).shape)  # (8, 3)

:doc:`../examples/rocket` and :doc:`../examples/minirocket` fit exactly such
models; the readout there is a ridge classifier on standardised features.
:doc:`../examples/deep_esn` reads a stack of reservoirs over every layer's
states with no pooling at all.

The linear layer
----------------

The readout does not have to come last. The closed-form and online learners
fit any real ``nn.Linear`` or :class:`~torchdyno.matrices.Dense` they are
pointed at, on whatever reaches it, with ``readout="path.to.module"`` taking a
name as ``model.get_submodule`` does. ``Dense`` is useful where the readout
should be a TorchDyno matrix, for instance to freeze or update it like any
other (see :doc:`learners`).

Writing a pooling
-----------------

A pooling is an ``nn.Module`` from ``(T, B, H)`` to ``(B, H)``. Skip NaN, as the
built-ins do, so that it works on padded batches; ``torch.nanmean`` and
``masked_fill`` do most of the work. :doc:`../advanced/custom-parts` covers the
other levels.
