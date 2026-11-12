========
Learners
========

A learner fits a model in place: ``learner.fit(model, train, val=None)`` returns
a :class:`~torchdyno.learners.FitResult` with the history of the fit, the
hyperparameters it selected (``best``), and the RNG state it started from
(``rng``). Any object with that method is a
:class:`~torchdyno.learners.Learner`, which is what lets a learner of your own
run beside the built-in ones. ``train`` and ``val`` are iterables of time-first
``(x, y)`` batches: a list of tuples is a loader.

A learner never assumes which parts a model has. It looks at what is frozen,
what is a readout and what is a cell, so the same model can be read out by
ridge regression or trained by backprop by choosing the learner alone.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Learner
     - Fits
     - Key arguments
   * - :class:`~torchdyno.learners.Ridge`
     - a linear readout in closed form
     - ``l2``, ``washout``, ``standardize``, ``score_fn``, ``class_weight``
   * - :class:`~torchdyno.learners.RLS`
     - a linear readout online, one sample at a time
     - ``l2``, ``washout``, ``forgetting``
   * - :class:`~torchdyno.learners.FORCE`
     - the readout of a :class:`~torchdyno.sequence.FeedbackCell`, while the
       loop runs
     - ``l2``, ``washout``, ``forgetting``
   * - :class:`~torchdyno.learners.Backprop`
     - every parameter that requires grad, by gradient descent
     - ``loss_fn``, ``optimizer``, ``lr``, ``epochs``, ``scheduler``, ``clip``
   * - :class:`~torchdyno.learners.IntrinsicPlasticity`
     - the gains and biases of tanh units, toward a Gaussian output
     - ``lr``, ``mu``, ``sigma``, ``epochs``
   * - :class:`~torchdyno.learners.QuantileBias`
     - the biases of random convolutions, from quantiles of their outputs
     - ``conv``
   * - :class:`~torchdyno.learners.Chain`
     - several learners, one after the other
     - the learners, in order

Two library-wide rules apply to every learner. **Freezing is
``requires_grad_(False)``**, and it never changes a part's initialisation;
learners that adapt a part directly, such as intrinsic plasticity, do so
whether or not it is frozen to backprop. And **NaN marks a step a sequence does
not have**: time-aligned rows whose features are all NaN are dropped from every
fit, loss and score, while rows that are NaN in only some features, or not
finite, raise an error rather than fit on garbage (see
:doc:`../advanced/variable-length`).

The models below share one small task and a reservoir factory.

.. code-block:: python

   import torch
   from torch import nn

   from torchdyno.learners import (
       FORCE,
       RLS,
       Backprop,
       Chain,
       IntrinsicPlasticity,
       Ridge,
       cache_features,
       evaluate,
   )
   from torchdyno.matrices import Dense, spectral, uniform
   from torchdyno.objectives import NRMSE, Accuracy, Regression
   from torchdyno.readout import Last
   from torchdyno.sequence import FeedbackCell, LeakyCell, Recurrent

   def reservoir(features, size=100, leak=1.0):
       return Recurrent(
           LeakyCell(
               input=Dense(features, size, init=uniform(1.0)),
               recurrent=Dense(size, size, init=spectral(uniform(1.0), 0.9)),
               leak=leak,
           )
       ).requires_grad_(False)

   torch.manual_seed(0)
   x = torch.randn(600, 4, 1)
   y = torch.roll(x, 5, dims=0)  # recall the input 5 steps back
   train, val = [(x[:400], y[:400])], [(x[400:], y[400:])]

Closed-form and online readouts
-------------------------------

:class:`~torchdyno.learners.Ridge`, :class:`~torchdyno.learners.RLS` and
:class:`~torchdyno.learners.FORCE` fit a real ``nn.Linear`` or
:class:`~torchdyno.matrices.Dense`: the model's last module by default, or the
one ``readout`` names, as ``model.get_submodule`` takes a name, wherever it
sits. They capture whatever reaches that module, so only the readout changes.
Time-aligned features are fitted at every step, after the washout; pooled ones
once per sequence; integer targets become one-hot. This is what makes frozen
reservoirs cheap to fit: no gradients, and the readout is the only part that
changes.

Ridge
~~~~~

Ridge keeps its sums in float64. Given several penalties it selects one with its
``score_fn``, which needs a ``mode`` (a :class:`~torchdyno.objectives.Metric`
has one, or pass ``mode=``): on ``val``, computing the validation features once
for the whole sweep, or by exact leave-one-out when there is no ``val``.
``standardize=True`` scales the features to unit variance, as ROCKET does, and
``class_weight`` weights each sample's error by its class, or balances the
classes with ``"balanced"``. The chosen penalty is in ``result.best``.

.. code-block:: python

   esn = nn.Sequential(reservoir(1), nn.Linear(100, 1))
   result = Ridge([1e-6, 1e-3, 1.0], washout=20, score_fn=NRMSE()).fit(esn, train, val)
   print(result.best)  # the chosen l2 and its validation score
   print(evaluate(esn, val, NRMSE(), washout=20))

For a classifier, pool first and the features are one row per sequence:

.. code-block:: python

   labels = torch.randint(3, (200,))
   series = torch.randn(30, 200, 1) + labels[None, :, None]
   data = [(series, labels)]
   classifier = nn.Sequential(reservoir(1), Last(), nn.Linear(100, 3))
   Ridge(1e-3, standardize=True, class_weight="balanced").fit(classifier, data)
   print(evaluate(classifier, data, Accuracy()))

.. note::

   ``washout`` is for time-aligned features, where a reservoir's first steps
   still depend on its initial state. Pooled features raise ``ValueError`` if a
   washout is set.

RLS
~~~

:class:`~torchdyno.learners.RLS` updates the readout one sample at a time, at
``O(H²)`` per update, and keeps its inverse-covariance state across fits, for
streams, until ``reset()``. One pass with ``forgetting=1`` gives the ridge
solution; a ``forgetting`` below one discounts old samples, so the readout
tracks a changing signal. Its bias is penalised like a weight, unlike Ridge's.

.. code-block:: python

   online = nn.Sequential(reservoir(1), nn.Linear(100, 1))
   rls = RLS(l2=1.0, washout=20, forgetting=0.9999)
   rls.fit(online, train)
   print(evaluate(online, val, NRMSE(), washout=20))
   rls.reset()  # forget the accumulated statistics before an unrelated fit

FORCE
~~~~~

:class:`~torchdyno.learners.FORCE` fits the readout of a
:class:`~torchdyno.sequence.FeedbackCell` while the model runs as it is,
feeding back its own output. The same recursive least squares updates the
readout at every step on the new state and the target, so the fed-back output
stays close to the target from the start. ``cell`` names the feedback cell
when the model has more than one.

.. code-block:: python

   wave = torch.sin(torch.arange(800.0) / 5)[:, None, None]
   readout = nn.Linear(100, 1)
   loop = nn.Sequential(
       Recurrent(
           FeedbackCell(
               LeakyCell(
                   input=Dense(1, 100, init=uniform(1.0)),
                   recurrent=Dense(100, 100, init=spectral(uniform(1.0), 1.2)),
                   leak=0.2,
               ),
               readout,
           ),
           carry=True,
       ),
       readout,
   ).requires_grad_(False)
   FORCE(l2=1.0, washout=50).fit(loop, [(torch.zeros(800, 1, 0), wave)])
   print(loop(torch.zeros(100, 1, 0)).shape)  # generates with no input

See :doc:`../examples/force` and :doc:`../examples/rls_forcing` for the full
loops. The difference between the two is whether the reservoir reads the true
signal (teacher forcing) or its own prediction while it is fitted.

Backprop
--------

:class:`~torchdyno.learners.Backprop` trains whatever requires grad, so freezing
a part with ``requires_grad_(False)`` decides what trains. Parts that are frozen
stay bit-identical. It needs a ``loss_fn``, for which a task supplies one.

Beyond that it takes:

- an ``optimizer`` by name (``"adam"`` or ``"sgd"``, with ``lr``) or as a
  callable given the ``(name, parameter)`` pairs that require grad, so it can
  group parameters by name;
- a ``scheduler``, stepped per epoch or per batch with ``schedule=``; a
  ``ReduceLROnPlateau`` is stepped on the validation score, or on the training
  loss without one;
- gradient clipping with ``clip``, and a ``washout`` of leading steps left out
  of the loss and the score;
- a ``callback(epoch, history, model)`` called after every epoch, which stops
  training by returning ``True``;
- with a ``score_fn`` and ``val``, a ``history["val_score"]`` and, with
  ``keep_best=True``, the weights of the best epoch rather than the last.

.. code-block:: python

   task = Regression()
   end_to_end = nn.Sequential(Recurrent(
       LeakyCell(input=Dense(1, 32), recurrent=Dense(32, 32), leak=0.5)
   ), nn.Linear(32, 1))
   result = Backprop(
       task.loss,
       lr=1e-2,
       epochs=3,
       score_fn=task.primary,
       washout=20,
       keep_best=True,
       clip=1.0,
   ).fit(end_to_end, train, val)
   print(result.history["train_loss"], result.history["val_score"])

Backprop through a long recurrence is costly, and ``carry=True`` on a
:class:`~torchdyno.sequence.Recurrent` with chunked batches gives truncated
backprop through time. Freeze the reservoir and train only the readout, and
backprop gives a gradient-based counterpart of Ridge; freeze nothing, and it
trains the whole model, as in :doc:`../examples/cornn` and
:doc:`../examples/cfc`.

.. warning::

   If everything is frozen, there is nothing to train. Freeze deliberately, and
   check ``[n for n, p in model.named_parameters() if p.requires_grad]``.

Adapting the parts
------------------

:class:`~torchdyno.learners.IntrinsicPlasticity` is unsupervised: it ignores the
targets and moves each tanh unit's gain and bias online so that its activation
approaches ``N(mu, sigma²)``. A cell takes part if it has a per-unit ``gain``
and ``bias``, a ``preactivation(u, h)`` and a tanh activation, as a
:class:`~torchdyno.sequence.LeakyCell` does. It changes only the gains and
biases, even if they are frozen. Fit the readout afterwards.
:doc:`../examples/esn_plasticity` shows the saturated units it repairs.

:class:`~torchdyno.learners.QuantileBias` is the equivalent for random
convolutions: it sets each kernel's bias from a quantile of its outputs on a
random training sequence, so that :class:`~torchdyno.readout.PPV` of each kernel
is close to a chosen share. Pair it with
:func:`~torchdyno.sequence.minirocket`; see :doc:`../examples/minirocket`.

Chaining
--------

:class:`~torchdyno.learners.Chain` runs learners on the same model in order,
giving each the same ``train`` and ``val``. Its result merges theirs, keyed
``"<index>.<name>"``. ``Chain(IntrinsicPlasticity(), Ridge(...))`` adapts a
reservoir and then reads it out; ``Chain(Ridge(...), Backprop(...))`` fine-tunes
from the closed-form solution.

.. code-block:: python

   esn = nn.Sequential(reservoir(1), nn.Linear(100, 1))
   result = Chain(
       IntrinsicPlasticity(lr=1e-3),
       Ridge([1e-6, 1e-3, 1.0], washout=20, score_fn=NRMSE()),
   ).fit(esn, train, val)
   print(result.best)  # keys such as "1.l2"
   print(evaluate(esn, val, NRMSE(), washout=20))

Caching features
----------------

When a frozen part of a model is shared by many fits,
:func:`~torchdyno.learners.cache_features` runs it once, in eval mode, and
returns a loader of its outputs for the rest of the model. Sweeping a readout's
hyperparameters then does not rerun the reservoir.

.. code-block:: python

   states = cache_features(esn[:-1], train)
   cached_val = cache_features(esn[:-1], val)
   for l2 in (1e-6, 1e-3, 1.0):
       Ridge(l2, washout=20).fit(nn.Sequential(esn[-1]), states)  # fits esn[-1]
       print(l2, evaluate(nn.Sequential(esn[-1]), cached_val, NRMSE(), washout=20))

.. note::

   Only cache a part that is deterministic and does not change between fits. A
   part another learner is still adapting, or a ``carry=True`` layer, makes the
   cache stale.

Writing a learner
-----------------

A learner is any object with ``fit(model, train, val=None) -> FitResult``; a
local rule fits in a few lines and chains with the rest. See
:doc:`../advanced/custom-parts`. Benchmarks take a learner the same way, so
what you write there runs across seeds and conditions without change (see
:doc:`../advanced/benchmarks`).
