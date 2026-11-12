SCN
===

Fixed block-diagonal modules coupled by a trained skew-symmetric matrix. The blocks are frozen, while the coupling, the input and the readout train. The model is an :class:`~torchdyno.sequence.EulerCell` run by :class:`~torchdyno.sequence.Recurrent`, trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- how :class:`~torchdyno.matrices.BlockDiagonal` and :class:`~torchdyno.matrices.SkewCoupling` split a state into modules and couple them,
- how to freeze one part of a model and train the rest,
- how to reach a part inside a model to freeze it.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.scn

It prints the accuracy on the test batches:

.. code-block:: text

   SCN test accuracy: 0.9709

The last digits can differ across machines and PyTorch versions.

The data
--------

``frequency_task`` from ``examples/data.py`` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches ``(60, B, 1)``. The example uses the train and test batches.

.. literalinclude:: ../../../../examples/data.py
   :language: python
   :pyobject: frequency_task

The model
---------

The recurrent matrix is :class:`~torchdyno.matrices.BlockDiagonal` with 8 blocks of 16 units, bounded by ``torch.tanh``. The coupling is :class:`~torchdyno.matrices.SkewCoupling` over the same blocks, with ``pairs="ring"`` coupling each block to the next. Both are matrices, level 1, and both go into a :class:`~torchdyno.sequence.LeakyField`, together with a :class:`~torchdyno.matrices.Dense` input. :class:`~torchdyno.sequence.EulerCell` integrates the field with a constant step of 0.1. :class:`~torchdyno.readout.Last` keeps the final state and ``nn.Linear`` maps it to the three classes.

.. literalinclude:: ../../../../examples/scn.py
   :language: python
   :start-at: seed_all(0)
   :end-before: scn[0]

Freezing the blocks
-------------------

``scn[0]`` is the :class:`~torchdyno.sequence.Recurrent` layer, ``.cell.field`` the field, and ``.recurrent`` the block-diagonal matrix. ``requires_grad_(False)`` freezes it: its values stay as initialised while the rest trains. Freezing never changes how a matrix is initialised.

.. literalinclude:: ../../../../examples/scn.py
   :language: python
   :start-at: scn[0]
   :end-before: Backprop(

Training
--------

:class:`~torchdyno.learners.Backprop` trains whatever still requires grad, here the coupling, the input matrix and the readout, with Adam at a learning rate of ``1e-2`` for 30 epochs. :func:`~torchdyno.learners.evaluate` scores the test batches with :class:`~torchdyno.objectives.Accuracy`.

.. literalinclude:: ../../../../examples/scn.py
   :language: python
   :start-at: Backprop(

What to try
-----------

- Remove the freezing line to train the blocks too, and compare the accuracy.
- Replace the ring with ``pairs=8`` or ``pairs=0.5``, which couples a number or a share of randomly chosen pairs.
- Freeze the coupling instead and train the blocks: ``scn[0].cell.field.coupling.requires_grad_(False)``.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/scn.py
      :language: python

Related concepts
----------------

- :doc:`../concepts/matrices`
- :doc:`../concepts/sequence`
- :doc:`../getting-started/train-end-to-end`
