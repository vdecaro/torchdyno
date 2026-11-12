MiniRocket
==========

MiniRocket's fixed kernel patterns, their biases set from quantiles of the kernels' outputs on the training series, PPV pooling and a ridge classifier. The kernels are a :class:`~torchdyno.sequence.RandomConv` drawn by :func:`~torchdyno.sequence.minirocket`, and two learners are chained: :class:`~torchdyno.learners.QuantileBias`, then :class:`~torchdyno.learners.Ridge`.

Use it when you want to understand:

- how :func:`~torchdyno.sequence.minirocket` plugs into :class:`~torchdyno.sequence.RandomConv` as its draw,
- how :class:`~torchdyno.learners.QuantileBias` sets biases from data,
- how :class:`~torchdyno.learners.Chain` runs two learners on one model.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.minirocket

It prints the accuracy on the test batches:

.. code-block:: text

   MiniRocket test accuracy: 1.0000

The last digits can differ across machines and PyTorch versions.

The data
--------

``frequency_task`` from ``examples/data.py`` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches ``(60, B, 1)``.

.. literalinclude:: ../../../../examples/data.py
   :language: python
   :pyobject: frequency_task

The model
---------

``draw=minirocket`` replaces the default ROCKET draw, the way a matrix takes an initialiser. MiniRocket has 84 fixed patterns at spread dilations, so ``kernels`` must be a multiple of 84; 9 996 is 84 times 119. The biases start at zero. A single :class:`~torchdyno.readout.PPV`, level 3, pools each kernel to one feature, so the ``nn.Linear`` readout has 9 996 inputs.

.. literalinclude:: ../../../../examples/minirocket.py
   :language: python
   :start-at: seed_all(0)
   :end-before: l2 = torch.logspace

Fitting
-------

:class:`~torchdyno.learners.Chain` runs its learners one after the other on the same model. :class:`~torchdyno.learners.QuantileBias` first sets each kernel's bias to minus a quantile of its outputs on a training series, so that its PPV is about one minus that quantile. :class:`~torchdyno.learners.Ridge` then standardises the features, picks its penalty among ten values on the validation batches, and fits the readout.

.. literalinclude:: ../../../../examples/minirocket.py
   :language: python
   :start-at: l2 = torch.logspace
   :end-before: print(

Scoring
-------

.. literalinclude:: ../../../../examples/minirocket.py
   :language: python
   :start-at: print(

What to try
-----------

- Drop :class:`~torchdyno.learners.QuantileBias` from the chain, leaving the biases at zero, to see what the data-driven biases add.
- Use fewer kernels, 84 times a whole number such as 2 520, and set the readout width to match.
- Compare with the :doc:`ROCKET <rocket>` example, which draws random biases and adds max pooling.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/minirocket.py
      :language: python

Related concepts
----------------

- :doc:`../concepts/sequence`
- :doc:`../concepts/readout`
- :doc:`../concepts/learners`
