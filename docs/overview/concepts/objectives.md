# Objectives

The fourth level says what a model is trained and scored for. A
[`Metric`][torchdyno.objectives.Metric] is a score, callable as
`metric(pred, target) -> float`, with a `mode` that says whether smaller
(`"min"`) or larger (`"max"`) is better, so that a learner can select a
hyperparameter on it. A [`Task`][torchdyno.objectives.Task] bundles a loss, the
metrics to report and the primary metric that selects. Nothing here depends on
a model, so the same objective scores a ridge readout and an end-to-end
network.

## Metrics

| Metric | Mode | Scores |
|---|---|---|
| [`Accuracy`][torchdyno.objectives.Accuracy] | max | the fraction of correct top-1 predictions |
| [`BalancedAccuracy`][torchdyno.objectives.BalancedAccuracy] | max | the mean over classes of each class's recall |
| [`F1`][torchdyno.objectives.F1] | max | the macro-averaged F1 score |
| [`MSE`][torchdyno.objectives.MSE] | min | the mean squared error |
| [`NRMSE`][torchdyno.objectives.NRMSE] | min | the RMSE of each output over its standard deviation, averaged |
| [`MemoryCapacity`][torchdyno.objectives.MemoryCapacity] | max | the sum of squared correlations of outputs that recall past inputs |

Classification metrics take `(..., C)` logits and `(...)` class indices.
Regression metrics take predictions and targets of the same shape, and raise
`ValueError` otherwise, because broadcasting a `(B, 1)` prediction against
a `(B,)` target would silently score every prediction against every target.

Which to choose:

- `Accuracy` is the default for balanced classes. On imbalanced data a
    majority-class predictor scores well on it; use `BalancedAccuracy` (a
    constant prediction scores `1/C`) or `F1`.
- `NRMSE` divides each output by its own standard deviation, so outputs of
    different scales average fairly, and a score of 1 is no better than
    predicting the mean. Use `MSE` when the scale matters.
- `MemoryCapacity` is for outputs `k` that recall the input `k` steps back,
    one per delay, and sums their squared correlations, so `K` outputs score at
    most `K`.

[`ConfusionMatrix`][torchdyno.objectives.ConfusionMatrix] is a score but not a `Metric`:
it returns the `(C, C)` counts, true class by row and predicted class by
column, and has no mode, so nothing can select on it.

```python
import torch

from torchdyno.objectives import (
    F1,
    MSE,
    NRMSE,
    Accuracy,
    BalancedAccuracy,
    ConfusionMatrix,
    MemoryCapacity,
)

logits, labels = torch.randn(32, 4), torch.randint(4, (32,))
print(Accuracy()(logits, labels), BalancedAccuracy()(logits, labels), F1()(logits, labels))
print(ConfusionMatrix()(logits, labels).sum())  # 32 predictions counted

pred, target = torch.randn(200, 3), torch.randn(200, 3)
print(MSE()(pred, target), NRMSE()(pred, target), Accuracy.mode, NRMSE.mode)
print(MemoryCapacity()(target + 0.1 * pred, target))  # close to 3
```

## Tasks

[`SequenceClassification`][torchdyno.objectives.SequenceClassification] (one label per sequence:
`(B, C)` logits against `(B,)` indices),
[`TimestepClassification`][torchdyno.objectives.TimestepClassification] (one per step: ``(T, B,
C)` against `(T, B)``) and [`Regression`][torchdyno.objectives.Regression]
(time-aligned `(T, B, O)` targets) are ready-made. Each has a `loss` (cross
entropy, or squared error), reports its metrics with ``task.metrics(pred,
target)` and exposes the metric that selects as `task.primary``.
`Regression` takes `primary="nrmse"` or `"mse"`.

```python
from torchdyno.objectives import Regression, SequenceClassification

task = SequenceClassification()
print(task.metrics(logits, labels), task.loss(logits, labels).item(), task.primary.mode)

regression = Regression(primary="mse")
y = torch.randn(50, 4, 3)
print(regression.metrics(y + 0.1, y), regression.primary)
```

Benchmarks hand one back as `data.task`, so `data.task.primary` is the
`score_fn` of a learner and `data.task.loss` the loss of
[`Backprop`][torchdyno.learners.Backprop] (see [Learners](learners.md)).

Any other task is built directly, without subclassing: a dict of metrics, the
name of the primary one, and optionally a loss.

```python
import torch.nn.functional as F

from torchdyno.objectives import Task

imbalanced = Task(
    {"f1": F1(), "balanced_accuracy": BalancedAccuracy(), "accuracy": Accuracy()},
    primary="f1",
    loss=F.cross_entropy,
)
print(imbalanced.metrics(logits, labels), imbalanced.loss(logits, labels))
```

A task without `loss` raises `NotImplementedError` when asked for one, which
is fine when only a closed-form learner fits the model. An unknown `primary`
raises `ValueError` at construction. To change a task's loss while keeping
its metrics, subclass and override `loss`.

## Writing a metric

A metric subclasses [`Metric`][torchdyno.objectives.Metric], sets `mode` and
implements `__call__`. Any plain function of `(pred, target)` also works as
a `score_fn`, but without a `mode` the learners need to be told which
direction is better (`mode="min"` on [`Ridge`][torchdyno.learners.Ridge] and
[`Backprop`][torchdyno.learners.Backprop]).

```python
from torchdyno.objectives import Metric

class MAE(Metric):
    mode = "min"

    def __call__(self, pred, target):
        return (pred - target).abs().mean().item()

print(MAE()(pred, target), Task({"mae": MAE()}, primary="mae").primary.mode)
```

## Scoring a model

[`evaluate()`][torchdyno.learners.evaluate] runs a model over a loader in eval mode,
without gradients, and scores all its predictions at once, with any
`score_fn`: a metric, a task's `metrics` or a `ConfusionMatrix`. It
restores the model's modes afterwards. Scoring all predictions together, not
batch by batch, matters for metrics that are not means, such as F1 or a
correlation.

Time-aligned `(T, B, O)` predictions are scored as one row per step, so
batches of different lengths combine. The steps a sequence does not have (NaN in
every feature) are left out, and so is an optional `washout` of leading
steps, which a reservoir needs while it forgets its initial state.

```python
from torch import nn

from torchdyno.learners import evaluate

model = nn.Linear(3, 3)
loader = [(torch.randn(100, 2, 3), torch.randn(100, 2, 3))]
print(evaluate(model, loader, NRMSE(), washout=10))
print(evaluate(model, loader, regression.metrics, washout=10))
```

!!! note

    `washout` applies to time-aligned outputs only. Asking for one on pooled
    `(B, O)` outputs raises `ValueError`, as does a washout that leaves no
    step.
