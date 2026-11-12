LRU
===

A linear complex diagonal recurrence run with a parallel scan, trained end to end. The recurrence is linear, so a position-wise ``nn.Linear`` and ``nn.GELU`` after it, as in the LRU paper, is what lets the model tell frequencies apart.

Use it when you want to understand:

- :class:`~torchdyno.sequence.LinearDiagonalCell` with a :class:`~torchdyno.sequence.RingDiagonal` transition,
- how :class:`~torchdyno.sequence.Scan` runs a linear cell in parallel over time,
- complex matrices, stored as real parts.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.lru

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/lru.py
      :language: python
