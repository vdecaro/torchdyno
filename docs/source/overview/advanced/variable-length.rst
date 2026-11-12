==============================
Sequences of different lengths
==============================

Sequences of different lengths share a batch padded with NaN: a timestep whose
features are all NaN is one the sequence does not have, at its end or at its
start. One tensor still flows through ``nn.Sequential``, and every part skips
the absent steps:

- :class:`~torchdyno.sequence.Recurrent` and :class:`~torchdyno.sequence.Scan`
  hold the state over an absent step and output NaN there;
- :class:`~torchdyno.sequence.RandomConv` reads absent steps as zeros and marks
  the outputs that depend on them NaN;
- every pooling skips NaN, so :class:`~torchdyno.readout.Last` returns each
  sequence's own last step;
- the learners and :func:`~torchdyno.learners.evaluate` leave absent steps of
  time-aligned outputs out of every fit, loss and score.

Features that are NaN or infinite in only some positions of a step are an
error, not padding: the learners raise rather than fit on them.

.. code-block:: python

   import torch

   from torchdyno.matrices import Dense
   from torchdyno.readout import Last
   from torchdyno.sequence import LeakyCell, Recurrent

   layer = Recurrent(LeakyCell(input=Dense(2, 16), recurrent=Dense(16, 16)))
   x = torch.randn(50, 2, 2)
   x[30:, 1] = torch.nan  # the second sequence has 30 steps
   states = layer(x)
   assert states[30:, 1].isnan().all()
   torch.testing.assert_close(Last()(states)[1], layer(x[:30, 1:])[-1, 0])

:func:`~torchdyno.benchmark.load_ucr` pads the variable-length datasets of the
UCR and UEA archives this way. :class:`~torchdyno.sequence.Reverse` runs a layer
backwards, so the padding at the end comes first, where a recurrent layer holds
its initial state over it.
