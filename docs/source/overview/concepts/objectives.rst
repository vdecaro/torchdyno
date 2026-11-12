==========
Objectives
==========

A :class:`~torchdyno.objectives.Metric` is a score, callable as
``metric(pred, target) -> float``, with a ``mode`` that says whether smaller
(``"min"``) or larger (``"max"``) is better, so that a learner can select a
hyperparameter on it. A :class:`~torchdyno.objectives.Task` bundles a loss,
the metrics to report, and the primary metric that selects.

Metrics
-------

.. list-table::
   :header-rows: 1

   * - Metric
     - Scores
   * - :class:`~torchdyno.objectives.Accuracy`
     - the fraction of correct top-1 predictions
   * - :class:`~torchdyno.objectives.BalancedAccuracy`
     - the mean over classes of each class's recall
   * - :class:`~torchdyno.objectives.F1`
     - the macro-averaged F1 score
   * - :class:`~torchdyno.objectives.MSE`
     - the mean squared error
   * - :class:`~torchdyno.objectives.NRMSE`
     - the RMSE of each output over its standard deviation, averaged
   * - :class:`~torchdyno.objectives.MemoryCapacity`
     - the sum of squared correlations of outputs that recall past inputs

:class:`~torchdyno.objectives.ConfusionMatrix` returns the matrix of counts
instead of a number. Classification metrics take ``(..., C)`` logits and
``(...)`` class indices; regression metrics take predictions and targets of the
same shape.

Tasks
-----

:class:`~torchdyno.objectives.SequenceClassification`,
:class:`~torchdyno.objectives.TimestepClassification` and
:class:`~torchdyno.objectives.Regression` are ready-made. Any other is built
directly, without subclassing:

.. code-block:: python

   import torch
   import torch.nn.functional as F

   from torchdyno.objectives import F1, Accuracy, BalancedAccuracy, Task

   imbalanced = Task(
       {"f1": F1(), "balanced_accuracy": BalancedAccuracy(), "accuracy": Accuracy()},
       primary="f1",
       loss=F.cross_entropy,
   )
   pred, target = torch.randn(32, 4), torch.randint(4, (32,))
   print(imbalanced.metrics(pred, target), imbalanced.loss(pred, target))

Scoring a model
---------------

:func:`~torchdyno.learners.evaluate` runs a model over a loader in eval mode
and scores all its predictions at once, with any ``score_fn``: a metric, a
task's ``metrics``, or a ``ConfusionMatrix``. Time-aligned ``(T, B, O)``
predictions are scored as one row per step, after an optional washout and
without the steps a sequence does not have.
