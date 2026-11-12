AdaDiag
=======

A sparse assembly of diagonal modules with an adaptive step, trained end to end. Each unit has a bounded diagonal self-weight, the modules are coupled by a skew-symmetric matrix, and a gate sets every unit's Euler step from the input and the state.

Use it when you want to understand:

- how :class:`~torchdyno.sequence.EulerCell` integrates a vector field,
- how a step size can be a module computed at every timestep.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.adadiag

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/adadiag.py
      :language: python
