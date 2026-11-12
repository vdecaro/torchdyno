S4D
===

S4D as a frozen feature extractor, pooled like ROCKET and read out by ridge. The modes are linear in the input, so the nonlinear PPV and max pooling is what separates the frequencies.

Use it when you want to understand:

- :class:`~torchdyno.sequence.ZOHDiagonal`, a zero-order-hold transition,
- how parts from different models, here S4D's modes and ROCKET's pooling, compose.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.s4d

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/s4d.py
      :language: python
