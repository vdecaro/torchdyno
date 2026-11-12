Deep echo state network
=======================

Three reservoirs run in series on the Lorenz system, read out over every layer's states.

Use it when you want to understand:

- :class:`~torchdyno.sequence.Stack`, which concatenates the output of every layer,
- why a plain ``nn.Sequential`` of the same reservoirs would read only the last one.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.deep_esn

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/deep_esn.py
      :language: python
