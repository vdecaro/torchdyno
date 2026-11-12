=============
Sequence maps
=============

A sequence map takes a time-first ``(T, B, F)`` sequence to a ``(T, B, H)``
one: a cell run over time by a layer, random convolutions, or a combination of
them.

Cells
-----

A :class:`~torchdyno.sequence.Cell` is one step of a recurrence,
``cell(u_t, h) -> h'`` on ``(B, state_size)`` states. The new state is also
the cell's output, so a layer only has to collect the states.

.. list-table::
   :header-rows: 1

   * - Cell
     - Update
     - Composes
   * - :class:`~torchdyno.sequence.LeakyCell`
     - a leaky integrator ``h' = (1 - a) h + a φ(g ⊙ (B u + W h) + c)``
     - echo state networks
   * - :class:`~torchdyno.sequence.EulerCell`
     - an Euler step ``h' = h + ε ⊙ f(u, h)`` of a vector field, e.g.
       :class:`~torchdyno.sequence.LeakyField`
     - AdaDiag, SCN
   * - :class:`~torchdyno.sequence.OscillatorCell`
     - coupled damped oscillators, positions and velocities side by side
     - coRNN
   * - :class:`~torchdyno.sequence.ClosedFormCell`
     - a closed-form continuous-time update from a backbone
     - CfC
   * - :class:`~torchdyno.sequence.LinearDiagonalCell`
     - ``h' = a ⊙ h + s ⊙ (B u)``, with ``(a, s)`` from
       :class:`~torchdyno.sequence.RingDiagonal` or
       :class:`~torchdyno.sequence.ZOHDiagonal`, or computed from the input
     - LRU, S4D
   * - :class:`~torchdyno.sequence.FeedbackCell`
     - any cell that also reads a readout of its own state
     - closed loops, FORCE

Steps such as EulerCell's ``step``, OscillatorCell's ``dt`` or ClosedFormCell's
``dt`` take a constant, a per-unit tensor kept as a buffer, or a callable
``step(u, h)`` computed at every timestep, such as a
:class:`~torchdyno.sequence.Gate`.

Layers
------

:class:`~torchdyno.sequence.Recurrent` runs a cell step by step;
:class:`~torchdyno.sequence.Scan` runs a linear cell by a parallel scan over
time, in ``ceil(log2 T)`` steps. Both start from zeros, or from ``h0``, and
return the state after every step.

.. code-block:: python

   import torch

   from torchdyno.matrices import Dense, spectral, uniform
   from torchdyno.sequence import LeakyCell, Recurrent

   layer = Recurrent(
       LeakyCell(
           input=Dense(2, 64, init=uniform(1.0)),
           recurrent=Dense(64, 64, init=spectral(uniform(1.0), 0.9)),
       )
   )
   x = torch.randn(100, 4, 2)
   states = layer(x)  # (100, 4, 64)
   torch.testing.assert_close(layer(x[50:], h0=states[49]), states[50:])

With ``carry=True`` a layer keeps its last state, detached, and the next call
starts from it, for truncated backprop through time or a stream fed in chunks;
``reset_state()`` starts over. To speed the loop up, compile the cell in place
with ``layer.cell.compile()`` rather than compiling the model.

Combining layers
----------------

.. list-table::
   :header-rows: 1

   * - Part
     - Does
   * - :class:`~torchdyno.sequence.Stack`
     - runs layers in series and concatenates every layer's output; with
       ``feed_input=True`` every layer also reads the input
   * - :class:`~torchdyno.sequence.Concat`
     - runs modules side by side on the same input and concatenates them
   * - :class:`~torchdyno.sequence.Sum`
     - runs modules side by side and adds them, e.g. a residual connection
   * - :class:`~torchdyno.sequence.Reverse`
     - runs a layer from the last step to the first
   * - :class:`~torchdyno.sequence.Diff`
     - the first difference over time

.. code-block:: python

   from torch import nn

   from torchdyno.sequence import Concat, Reverse, Stack, Sum

   def reservoir(features):
       return Recurrent(LeakyCell(input=Dense(features, 64), recurrent=Dense(64, 64)))

   bidirectional = Concat(reservoir(2), Reverse(reservoir(2)))  # (T, B, 128)
   deep = Stack(reservoir(2), reservoir(64), reservoir(64))  # (T, B, 192)
   residual = Sum(nn.Linear(2, 64), reservoir(2))  # (T, B, 64)
   print([part(x).shape[-1] for part in (bidirectional, deep, residual)])

Random convolutions
-------------------

:class:`~torchdyno.sequence.RandomConv` holds fixed random kernels as buffers
and maps ``(T, B, C)`` to ``(T, B, kernels)``. Like a matrix takes an
initialiser, it takes the kernels from a draw:
:func:`~torchdyno.sequence.rocket` by default, or
:func:`~torchdyno.sequence.minirocket`, whose biases are then fitted by
:class:`~torchdyno.learners.QuantileBias`. A kernel without padding produces
fewer outputs than the input has steps; the positions it does not produce are
NaN, which the pooling skips.

.. code-block:: python

   from torchdyno.sequence import RandomConv, minirocket

   rocket = RandomConv(channels=2, length=100, kernels=1_000)
   mini = RandomConv(channels=2, length=100, kernels=84 * 12, draw=minirocket)
   print(rocket(x).shape, mini(x).shape)  # (100, 4, 1000) (100, 4, 1008)
