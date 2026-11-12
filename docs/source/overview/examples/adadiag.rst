AdaDiag
=======

A sparse assembly of diagonal modules with an adaptive step, trained end to end. Each unit has a bounded diagonal self-weight, the modules are coupled by a skew-symmetric matrix, and a gate sets every unit's Euler step from the input and the state. The model is trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- how :class:`~torchdyno.sequence.EulerCell` and :class:`~torchdyno.sequence.LeakyField` compose a continuous-time model,
- how :class:`~torchdyno.matrices.Diagonal` and :class:`~torchdyno.matrices.SkewCoupling` structure the recurrence,
- how :class:`~torchdyno.sequence.Gate` makes the step size adaptive.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.adadiag

It prints the accuracy on the test batches:

.. code-block:: text

   AdaDiag test accuracy: 1.0000

The last digits can differ across machines and PyTorch versions.

The data
--------

``frequency_task`` builds noisy sine waves of three frequencies, and the class of a wave is its frequency. It returns train, validation and test lists of time-first batches, ``x`` of shape ``(60, B, 1)`` and integer labels ``y`` of shape ``(B,)``. This example does not need the validation set.

.. literalinclude:: ../../../../examples/data.py
   :language: python
   :pyobject: frequency_task

The model
---------

The cell is built from a vector field and a step. :class:`~torchdyno.sequence.LeakyField` combines three matrices, level 1. The recurrent matrix is :class:`~torchdyno.matrices.Diagonal`, a self-weight per unit, bounded by ``torch.tanh`` so it stays in ``(-1, 1)``. :class:`~torchdyno.matrices.SkewCoupling` couples 8 blocks of 16 units, with 8 pairs of blocks coupled, and stores only those pairs; the dense coupling ``C - Cᵀ`` stays skew-symmetric during training. The input matrix is a :class:`~torchdyno.matrices.Dense` drawn from :func:`~torchdyno.matrices.normal`.

:class:`~torchdyno.sequence.EulerCell` integrates the field with an explicit Euler step. Here the step is a :class:`~torchdyno.sequence.Gate`, a per-unit step size between ``1e-5`` and ``0.2`` driven by the input and the state. :class:`~torchdyno.sequence.Recurrent` runs the cell over time, and :class:`~torchdyno.readout.Last`, level 3, keeps the final state. An ``nn.Linear`` maps it to the three classes.

.. literalinclude:: ../../../../examples/adadiag.py
   :language: python
   :start-at: seed_all(0)
   :end-before: Backprop(

Training
--------

:class:`~torchdyno.learners.Backprop` trains every parameter that requires grad, with Adam by default, for 30 epochs, minimising the cross-entropy loss. :func:`~torchdyno.learners.evaluate` then scores the test batches with :class:`~torchdyno.objectives.Accuracy`, a metric of the objectives level.

.. literalinclude:: ../../../../examples/adadiag.py
   :language: python
   :start-at: Backprop(

What to try
-----------

- Use ``pairs="ring"`` in :class:`~torchdyno.matrices.SkewCoupling` to couple each block to its neighbour, as the :doc:`SCN <scn>` example does.
- Replace the gate with a constant step, ``step=0.1``, to see what the adaptive step adds.
- Change ``bounds`` of the gate, or the ``bound`` of the diagonal, and compare the accuracy.
- Run the three training conditions side by side in :doc:`conditions`.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/adadiag.py
      :language: python

Related concepts
----------------

- :doc:`../concepts/matrices`
- :doc:`../concepts/sequence`
- :doc:`../concepts/readout`
- :doc:`../getting-started/train-end-to-end`
