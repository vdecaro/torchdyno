========================
Train a model end to end
========================

The same composition style builds models trained by backprop. Here an
assembly of diagonal modules with an adaptive step, the AdaDiag model, tells
noisy sine waves apart by their frequency.

The data
--------

The classes are three frequencies. Each sequence is a ``(T, F) = (60, 1)``
series, stacked into batches of ``(60, B, 1)`` with one label per sequence.

.. code-block:: python

   import math

   import torch
   from torch import nn

   from torchdyno import seed_all

   seed_all(0)
   t = torch.linspace(0, 6 * math.pi, 60)
   labels = torch.randint(3, (384,))
   phase = 2 * math.pi * torch.rand(384)
   x = torch.sin((1 + labels) * t[:, None] + phase) + 0.3 * torch.randn(60, 384)
   x = x[..., None]  # (T, B, F) = (60, 384, 1)
   train = [(x[:, i : i + 64], labels[i : i + 64]) for i in range(0, 256, 64)]
   test = [(x[:, 256:], labels[256:])]

The model
---------

:class:`~torchdyno.sequence.EulerCell` takes Euler steps of an ODE's vector
field. :class:`~torchdyno.sequence.LeakyField` is that field, given a bounded
:class:`~torchdyno.matrices.Diagonal` self-weight per unit and a
:class:`~torchdyno.matrices.SkewCoupling` between modules of 16 units; a
:class:`~torchdyno.sequence.Gate` computes each unit's step from the input and
the state. :class:`~torchdyno.readout.Last` keeps the last state of each
sequence, so the classifier reads one ``(B, H)`` row per sequence.

.. code-block:: python

   from torchdyno.matrices import Dense, Diagonal, SkewCoupling, normal
   from torchdyno.readout import Last
   from torchdyno.sequence import EulerCell, Gate, LeakyField, Recurrent

   H = 16 * 4
   model = nn.Sequential(
       Recurrent(
           EulerCell(
               LeakyField(
                   input=Dense(1, H, init=normal(H**-0.5)),
                   recurrent=Diagonal(H, bound=torch.tanh),
                   coupling=SkewCoupling([16] * 4, pairs=4),
                   activation=torch.relu,
               ),
               step=Gate(1, H, bounds=(1e-5, 0.2)),
           )
       ),
       Last(),
       nn.Linear(H, 3),
   )

Train and score
---------------

:class:`~torchdyno.learners.Backprop` trains every parameter that requires
grad. Each learner returns a :class:`~torchdyno.learners.FitResult`, here with
the training loss of every epoch.

.. code-block:: python

   from torchdyno.learners import Backprop, evaluate
   from torchdyno.objectives import Accuracy

   result = Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=20).fit(model, train)
   print(f"loss {result.history['train_loss'][0]:.2f} -> {result.history['train_loss'][-1]:.2f}")
   print(f"test accuracy: {evaluate(model, test, Accuracy()):.2f}")

Where to go next
----------------

Freeze a part with ``requires_grad_(False)`` and only the rest trains, as the
:doc:`SCN example <../examples/scn>` does with its blocks. Swap the cell for a
:class:`~torchdyno.sequence.LinearDiagonalCell` run by
:class:`~torchdyno.sequence.Scan` and the same model becomes an LRU.
