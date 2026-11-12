Deep echo state network
=======================

Three reservoirs run in series on the Lorenz system, and a single ridge readout reads the states of all three. Each reservoir is a :class:`~torchdyno.sequence.Recurrent` over a :class:`~torchdyno.sequence.LeakyCell`, and :class:`~torchdyno.sequence.Stack` chains them. Only the readout is fitted.

Use it when you want to understand:

- how :class:`~torchdyno.sequence.Stack` runs layers in series and concatenates their outputs,
- why a deep reservoir reads out over every layer rather than only the last,
- how a helper function builds the same reservoir at different sizes.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.deep_esn

It prints the NRMSE on the evaluation segment:

.. code-block:: text

   deep ESN test NRMSE: 5.88e-04

The last digits can differ across machines and PyTorch versions.

The reservoir helper
--------------------

A function builds one reservoir of a given input and state size, so the three layers differ only in their shapes. The input matrix is drawn from ``uniform(0.5)``, the recurrent matrix is rescaled to a spectral radius of 0.9, and the leak is 0.5. These are the matrices, level 1, inside a cell and a layer of the sequence level, level 2.

.. literalinclude:: ../../../../examples/deep_esn.py
   :language: python
   :pyobject: reservoir

The model
---------

The first reservoir reads the three Lorenz variables. The second reads the states of the first, and the third reads the second. :class:`~torchdyno.sequence.Stack` returns all three outputs side by side, ``(T, B, 600)``, which is the width of the readout. A plain ``nn.Sequential`` of the same reservoirs would pass only the last one to the readout. The stack is frozen as a whole.

.. literalinclude:: ../../../../examples/deep_esn.py
   :language: python
   :start-at: seed_all(0)
   :end-before: Ridge(

Fitting and scoring
-------------------

A single penalty needs no validation loader, so :class:`~torchdyno.learners.Ridge` fits on the training loader alone, after the washout. :func:`~torchdyno.learners.evaluate` then scores the evaluation loader with the dataset's primary metric.

.. literalinclude:: ../../../../examples/deep_esn.py
   :language: python
   :start-at: Ridge(

What to try
-----------

- Pass ``feed_input=True`` to :class:`~torchdyno.sequence.Stack`, so every layer after the first also reads the input. The deeper layers then take ``inputs + size`` features, so change their input sizes in the helper calls.
- Swap the stack for a ``nn.Sequential`` of the same reservoirs, with ``nn.Linear(200, 3)``, to compare reading the last layer only.
- Give each layer its own leak or spectral radius, by passing them as arguments of ``reservoir``.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/deep_esn.py
      :language: python

Related concepts
----------------

- :doc:`../getting-started/first-reservoir`
- :doc:`../concepts/sequence`
- :doc:`../concepts/learners`
