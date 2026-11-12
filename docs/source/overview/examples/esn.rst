Echo state network
==================

A frozen leaky reservoir with a sparse recurrent matrix forecasts the Lorenz system, read out by ridge regression. The reservoir is a :class:`~torchdyno.sequence.LeakyCell` run by :class:`~torchdyno.sequence.Recurrent`, followed by an ``nn.Linear`` readout. Only the readout is fitted, in closed form, on a Lorenz trajectory cut into train, validation and evaluation segments.

Use it when you want to understand:

- how a reservoir composes from a cell and two matrices,
- how ``torch.nn.init`` functions such as ``sparse_`` work as initialisers,
- how ridge regression selects its penalty on a validation split after a washout.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.esn

It prints the normalised root mean squared error (NRMSE) on the evaluation segment:

.. code-block:: text

   ESN test NRMSE: 5.45e-06

The last digits can differ across machines and PyTorch versions.

The data
--------

:func:`~torchdyno.benchmark.lorenz_benchmark` integrates the Lorenz system and returns a dataset with ``train_loader``, ``val_loader`` and ``eval_loader``, each an iterable of time-first ``(x, y)`` batches of shape ``(T, B, 3)``. With ``target_delay=5`` the target at step ``t`` is the state at ``t + 5``, so the model forecasts five steps ahead. ``washout=100`` is the number of leading steps to leave out of fitting and scoring, while the reservoir forgets its zero initial state. ``seed_all`` seeds every random number generator, so the run repeats.

.. literalinclude:: ../../../../examples/esn.py
   :language: python
   :start-at: seed_all(0)
   :end-before: esn = nn.Sequential(

The model
---------

The reservoir spans two levels. The matrices, level 1, are :class:`~torchdyno.matrices.Dense` maps. The input matrix draws its weights and its bias from :func:`~torchdyno.matrices.uniform`. The recurrent matrix is drawn by an initialiser that wraps ``nn.init.sparse_`` in :func:`~torchdyno.matrices.spectral`, which rescales the result to a spectral radius of 0.9. Any callable that fills a tensor in place is an initialiser, so a ``torch.nn.init`` function fits without an adapter. The sequence layer, level 2, is :class:`~torchdyno.sequence.LeakyCell` run by :class:`~torchdyno.sequence.Recurrent`, which returns the state at every step, ``(T, B, 500)``. ``requires_grad_(False)`` freezes the reservoir; the readout is a plain ``nn.Linear``.

.. literalinclude:: ../../../../examples/esn.py
   :language: python
   :start-at: esn = nn.Sequential(
   :end-before: l2 = torch.logspace

Fitting
-------

:class:`~torchdyno.learners.Ridge` fits the last module of the model on whatever reaches it. Given a list of penalties, it fits each, scores it on the validation loader with ``score_fn`` and keeps the best. ``data.task.primary`` is the metric the dataset's task selects on, here NRMSE.

.. literalinclude:: ../../../../examples/esn.py
   :language: python
   :start-at: l2 = torch.logspace
   :end-before: print(

Scoring
-------

:func:`~torchdyno.learners.evaluate` runs the model on the evaluation loader and applies the metric, again after the washout.

.. literalinclude:: ../../../../examples/esn.py
   :language: python
   :start-at: print(

What to try
-----------

- Replace the sparse recurrent matrix with a ring: ``init=spectral(cycle(1.0), 0.9)``, using :func:`~torchdyno.matrices.cycle`.
- Raise ``target_delay`` to see the forecast degrade, and widen the grid of penalties to see which one the validation loader picks.
- Print the fit result, ``Ridge(...).fit(...).best``, to read the chosen penalty and its validation score.
- Lower the leak of the cell, ``leak=0.3``, to give the reservoir a longer memory.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/esn.py
      :language: python

Related concepts
----------------

- :doc:`../getting-started/first-reservoir`
- :doc:`../concepts/matrices`
- :doc:`../concepts/sequence`
- :doc:`../concepts/learners`
