Online readout
==============

Recursive least squares fits a reservoir's readout online while the reservoir reads the true previous Lorenz state; then the prediction is fed back as the next input, one step at a time.

Use it when you want to understand:

- how :class:`~torchdyno.learners.RLS` fits a readout one sample at a time,
- how a layer continues from a previous state with ``h0``.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.rls_forcing

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/rls_forcing.py
      :language: python
