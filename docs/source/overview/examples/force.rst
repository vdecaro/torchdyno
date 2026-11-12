FORCE
=====

A reservoir with no input generates a two-tone wave. Its readout is fed back into it, and FORCE fits the readout at every step while the loop runs.

Use it when you want to understand:

- :class:`~torchdyno.sequence.FeedbackCell`, which closes a loop through a readout,
- :class:`~torchdyno.learners.FORCE`, which trains that readout in the loop,
- how a carried state lets the trained loop keep generating.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.force

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/force.py
      :language: python
