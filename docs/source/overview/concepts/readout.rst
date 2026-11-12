=======
Readout
=======

The readout reduces a ``(T, B, H)`` sequence to ``(B, H)`` features by pooling
over time; a plain ``nn.Linear`` then maps them to outputs. For time-aligned
outputs, such as a forecast at every step, leave the pooling out and the
``nn.Linear`` acts at every step.

.. list-table::
   :header-rows: 1

   * - Pooling
     - Each feature becomes
   * - :class:`~torchdyno.readout.Last`
     - its value at each sequence's last step
   * - :class:`~torchdyno.readout.Mean`
     - its mean over time
   * - :class:`~torchdyno.readout.Max`
     - its maximum over time
   * - :class:`~torchdyno.readout.PPV`
     - the proportion of positive values (ROCKET)
   * - :class:`~torchdyno.readout.MPV`
     - the mean of the positive values (MultiRocket)
   * - :class:`~torchdyno.readout.MIPV`
     - the mean index of the positive values (MultiRocket)
   * - :class:`~torchdyno.readout.LSPV`
     - the longest stretch of positive values (MultiRocket)

Every pooling skips the positions that are NaN: those a convolution did not
produce, and the steps a shorter sequence does not have (see
:doc:`../advanced/variable-length`). Pool one sequence several ways with
:class:`~torchdyno.sequence.Concat`:

.. code-block:: python

   import torch
   from torch import nn

   from torchdyno.readout import LSPV, MPV, PPV, Max
   from torchdyno.sequence import Concat, RandomConv

   multirocket = nn.Sequential(
       RandomConv(channels=1, length=60, kernels=500),
       Concat(PPV(), Max(), MPV(), LSPV()),
       nn.Linear(4 * 500, 3),
   )
   print(multirocket(torch.randn(60, 8, 1)).shape)  # (8, 3)

A readout does not have to come last: the closed-form and online learners fit
any real ``nn.Linear`` or :class:`~torchdyno.matrices.Dense` they are pointed
at, on whatever reaches it (see :doc:`learners`).
