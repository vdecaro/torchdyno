LRU
===

A linear complex diagonal recurrence run with a parallel scan, trained end to end. The model is a :class:`~torchdyno.sequence.LinearDiagonalCell` run by :class:`~torchdyno.sequence.Scan`, followed by a position-wise nonlinearity, :class:`~torchdyno.readout.Mean` pooling and a readout, trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- how :class:`~torchdyno.sequence.RingDiagonal` parametrises complex eigenvalues in a ring,
- how :class:`~torchdyno.sequence.Scan` replaces the step-by-step loop by a parallel scan,
- why a linear recurrence needs a nonlinearity after it.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.lru

It prints the accuracy on the test batches:

.. code-block:: text

   LRU test accuracy: 1.0000

The last digits can differ across machines and PyTorch versions.

The data
--------

``frequency_task`` from ``examples/data.py`` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches ``(60, B, 1)``.

.. literalinclude:: ../../../../examples/data.py
   :language: python
   :pyobject: frequency_task

The model
---------

The cell has ``N = 64`` complex modes. The input matrix is a :class:`~torchdyno.matrices.Dense` with ``dtype=torch.cfloat``, a complex matrix, level 1. :class:`~torchdyno.sequence.RingDiagonal` is the transition: it draws each eigenvalue with a modulus between ``r_min`` and ``r_max`` and a phase up to ``max_phase``, so every mode is stable and decays at its own rate. Because the recurrence is linear in the state, :class:`~torchdyno.sequence.Scan` evaluates it in about ``log2 T`` parallel steps instead of ``T`` sequential ones.

The scan returns the real and imaginary parts side by side, ``2 * N`` features per step. The recurrence is linear, so on its own the model could not tell frequencies apart; a position-wise ``nn.Linear`` and ``nn.GELU`` supply the nonlinearity, and :class:`~torchdyno.readout.Mean`, level 3, averages over time before the readout.

.. literalinclude:: ../../../../examples/lru.py
   :language: python
   :start-at: seed_all(0)
   :end-before: Backprop(

Training and scoring
--------------------

:class:`~torchdyno.learners.Backprop` trains every parameter with Adam, at ``1e-2``, for 30 epochs on the cross-entropy loss. :func:`~torchdyno.learners.evaluate` scores the test batches with :class:`~torchdyno.objectives.Accuracy`.

.. literalinclude:: ../../../../examples/lru.py
   :language: python
   :start-at: Backprop(

What to try
-----------

- Swap :class:`~torchdyno.sequence.RingDiagonal` for :class:`~torchdyno.sequence.ZOHDiagonal`, as the :doc:`S4D <s4d>` example does.
- Narrow the ring, ``r_min=0.9``, to give every mode a long memory, and compare the accuracy.
- Replace :class:`~torchdyno.sequence.Scan` by :class:`~torchdyno.sequence.Recurrent` over the same cell to check that both give the same states, only faster or slower.
- Pool with :class:`~torchdyno.readout.Last` instead of :class:`~torchdyno.readout.Mean`.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/lru.py
      :language: python

Related concepts
----------------

- :doc:`../concepts/matrices`
- :doc:`../concepts/sequence`
- :doc:`../concepts/readout`
- :doc:`../getting-started/train-end-to-end`
