MiniRocket
==========

MiniRocket's fixed kernel patterns, their biases set from quantiles of the kernels' outputs on the training series, PPV pooling and a ridge classifier.

Use it when you want to understand:

- the :func:`~torchdyno.sequence.minirocket` draw,
- :class:`~torchdyno.learners.QuantileBias` chained before the ridge readout.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.minirocket

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/minirocket.py
      :language: python
