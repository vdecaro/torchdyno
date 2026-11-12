Intrinsic plasticity
====================

A strong input saturates about half of a reservoir's units. The same reservoir is read out as initialised, and after intrinsic plasticity has adapted its gains and biases toward unsaturated outputs.

Use it when you want to understand:

- how :class:`~torchdyno.learners.IntrinsicPlasticity` adapts a reservoir without labels,
- how two learners chain on the same model.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.esn_plasticity

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/esn_plasticity.py
      :language: python
