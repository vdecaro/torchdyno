Readout, frozen, trained
========================

One AdaDiag model run three ways on the same seeds: only the readout fitted by ridge; the same parts frozen, with a linear encoder in front and the readout trained by backprop; and everything trained.

Use it when you want to understand:

- :func:`~torchdyno.benchmark.run_conditions` and :func:`~torchdyno.benchmark.conditions_table`.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.conditions

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/conditions.py
      :language: python
