==========
Benchmarks
==========

:func:`~torchdyno.benchmark.run` fits a fresh model for every seed of a
:class:`~torchdyno.benchmark.BenchmarkSpec` and reports each metric of the
dataset's task as a mean and a standard deviation, as
:doc:`../getting-started/benchmark-across-seeds` shows.

Datasets
--------

A :class:`~torchdyno.benchmark.BenchmarkDataset` holds the train, evaluation
and optional validation loaders, the task, and the washout. Any iterable of
``(x, y)`` batches is a loader. These are built in:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Task
   * - :func:`~torchdyno.benchmark.lorenz_benchmark`
     - forecast the Lorenz system
   * - :func:`~torchdyno.benchmark.mackey_glass_benchmark`
     - forecast the Mackey-Glass delay equation
   * - :func:`~torchdyno.benchmark.narma_benchmark`
     - predict the output of a NARMA system from its random input
   * - :func:`~torchdyno.benchmark.memory_capacity_benchmark`
     - recall the input of each of the last steps, scored by memory capacity
   * - :func:`~torchdyno.benchmark.ucr_benchmark`
     - classify a UCR or UEA dataset read from local files

The synthetic ones are cut into contiguous train, validation and evaluation
segments. :func:`~torchdyno.benchmark.ucr_benchmark` reads the archives'
``.tsv`` and ``.ts`` files, given the folder the archive was extracted to:

.. code-block:: python

   from torchdyno.benchmark import ucr_benchmark

   data = ucr_benchmark("~/data/Univariate_arff", "GunPoint", batch_size=32, val=0.2)

Readout, frozen, trained
------------------------

:func:`~torchdyno.benchmark.run_conditions` runs one model three ways on the
same seeds, data and initialisation:

- ``readout``: everything before the readout frozen, the readout fitted by
  ridge regression;
- ``frozen``: the same parts frozen, a linear encoder in front and the readout
  trained by backprop;
- ``trained``: the whole model trained by backprop.

:func:`~torchdyno.benchmark.conditions_table` lays the results side by side, as
in the :doc:`conditions example <../examples/conditions>`.

Reproducibility
---------------

:func:`~torchdyno.seed_all` seeds Python, torch and any generators given, such
as a ``DataLoader``'s, and by default asks torch for deterministic algorithms;
:func:`~torchdyno.benchmark.run` calls it before every build. Every
:class:`~torchdyno.learners.FitResult` keeps the RNG state its fit started
from, which :func:`~torchdyno.set_rng_state` restores to replay the fit.
