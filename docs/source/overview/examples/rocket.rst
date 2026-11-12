ROCKET
======

Ten thousand random convolution kernels, PPV and max pooling, and a ridge classifier on standardised features, as in the ROCKET paper.

Use it when you want to understand:

- :class:`~torchdyno.sequence.RandomConv` and the :func:`~torchdyno.sequence.rocket` draw,
- how :class:`~torchdyno.sequence.Concat` pools one sequence two ways.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.rocket

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/rocket.py
      :language: python
