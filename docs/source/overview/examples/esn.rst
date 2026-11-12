Echo state network
==================

A frozen leaky reservoir with a sparse recurrent matrix forecasts the Lorenz system, read out by ridge regression.

Use it when you want to understand:

- how a reservoir composes from a cell and two matrices,
- how ``torch.nn.init`` functions such as ``sparse_`` work as initialisers,
- how ridge regression selects its penalty on a validation split after a washout.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.esn

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/esn.py
      :language: python
