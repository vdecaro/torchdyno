ROCKET
======

Ten thousand random convolution kernels, PPV and max pooling, and a ridge classifier on standardised features. The kernels are a :class:`~torchdyno.sequence.RandomConv`, frozen by construction, and the readout is fitted by ridge on a synthetic task of noisy sine waves classified by their frequency.

Use it when you want to understand:

- how :class:`~torchdyno.sequence.RandomConv` draws and holds a bank of fixed kernels,
- how :class:`~torchdyno.sequence.Concat` joins PPV and max pooling,
- how ridge selects its penalty on a validation split with standardised features.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.rocket

It prints the accuracy on the test batches:

.. code-block:: text

   ROCKET test accuracy: 1.0000

The last digits can differ across machines and PyTorch versions.

The data
--------

``frequency_task`` from ``examples/data.py`` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches ``(60, B, 1)``.

.. literalinclude:: ../../../../examples/data.py
   :language: python
   :pyobject: frequency_task

The model
---------

:class:`~torchdyno.sequence.RandomConv` takes the number of ``channels``, the series ``length`` and the number of ``kernels``, and draws them with :func:`~torchdyno.sequence.rocket`, its default. Kernels are kept as buffers, so nothing needs freezing. A kernel without padding leaves NaN at the steps it does not cover, which the pooling skips. :class:`~torchdyno.readout.PPV` and :class:`~torchdyno.readout.Max`, level 3, are run side by side by :class:`~torchdyno.sequence.Concat`, giving two features per kernel, 20 000 in all, for the ``nn.Linear`` readout.

.. literalinclude:: ../../../../examples/rocket.py
   :language: python
   :start-at: seed_all(0)
   :end-before: l2 = torch.logspace

Fitting and scoring
-------------------

:class:`~torchdyno.learners.Ridge` standardises the features, tries ten penalties from ``1e-3`` to ``1e3``, and keeps the one with the best accuracy on the validation batches. :func:`~torchdyno.learners.evaluate` scores the test batches with :class:`~torchdyno.objectives.Accuracy`.

.. literalinclude:: ../../../../examples/rocket.py
   :language: python
   :start-at: l2 = torch.logspace

What to try
-----------

- Reduce ``kernels`` to 1 000, and change the readout to ``nn.Linear(2_000, 3)``, to see how little accuracy this task needs.
- Pool with :class:`~torchdyno.readout.PPV` alone, and compare.
- Swap the kernel draw for MiniRocket's: ``draw=minirocket``, as in the :doc:`MiniRocket <minirocket>` example.
- Pass ``class_weight="balanced"`` to :class:`~torchdyno.learners.Ridge` on an unbalanced dataset.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/rocket.py
      :language: python

Related concepts
----------------

- :doc:`../concepts/sequence`
- :doc:`../concepts/readout`
- :doc:`../concepts/learners`
- :doc:`../advanced/benchmarks`
