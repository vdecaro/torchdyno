SCN
===

Fixed block-diagonal modules coupled by a trained skew-symmetric matrix: the blocks are frozen, while the coupling, the input and the readout train.

Use it when you want to understand:

- how ``requires_grad_(False)`` freezes one part of a model trained by backprop,
- :class:`~torchdyno.matrices.BlockDiagonal` and :class:`~torchdyno.matrices.SkewCoupling`.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.scn

.. dropdown:: Example code
   :open:

   .. literalinclude:: ../../../../examples/scn.py
      :language: python
