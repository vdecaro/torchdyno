========
Learners
========

A learner fits a model in place: ``learner.fit(model, train, val=None)`` returns
a :class:`~torchdyno.learners.FitResult` with the history of the fit, the
hyperparameters it selected, and the RNG state it started from. Any object with
that method is a :class:`~torchdyno.learners.Learner`.

.. list-table::
   :header-rows: 1

   * - Learner
     - Fits
   * - :class:`~torchdyno.learners.Ridge`
     - a linear readout in closed form
   * - :class:`~torchdyno.learners.RLS`
     - a linear readout online, one sample at a time
   * - :class:`~torchdyno.learners.FORCE`
     - the readout of a :class:`~torchdyno.sequence.FeedbackCell`, while the
       loop runs
   * - :class:`~torchdyno.learners.Backprop`
     - every parameter that requires grad, by gradient descent
   * - :class:`~torchdyno.learners.IntrinsicPlasticity`
     - the gains and biases of tanh units, toward a Gaussian output
   * - :class:`~torchdyno.learners.QuantileBias`
     - the biases of random convolutions, from quantiles of their outputs
   * - :class:`~torchdyno.learners.Chain`
     - several learners, one after the other

Closed-form and online readouts
-------------------------------

:class:`~torchdyno.learners.Ridge`, :class:`~torchdyno.learners.RLS` and
:class:`~torchdyno.learners.FORCE` fit a real ``nn.Linear`` or
:class:`~torchdyno.matrices.Dense`: the model's last module by default, or the
one ``readout`` names, wherever it sits. They capture whatever reaches it, so
only the readout changes. Time-aligned features are fitted at every step, after
the washout; pooled ones once per sequence; integer targets become one-hot.

Ridge keeps its sums in float64. Given several penalties it selects one with
its ``score_fn``: on ``val``, computing the validation features once for the
whole sweep, or by exact leave-one-out without ``val``. ``standardize=True``
scales the features to unit variance, as ROCKET does, and ``class_weight``
weights each sample's error by its class, or balances the classes.

.. code-block:: python

   import torch
   from torch import nn

   from torchdyno.learners import Chain, IntrinsicPlasticity, Ridge, evaluate
   from torchdyno.matrices import Dense, spectral, uniform
   from torchdyno.objectives import NRMSE
   from torchdyno.sequence import LeakyCell, Recurrent

   torch.manual_seed(0)
   x = torch.randn(600, 4, 1)
   y = torch.roll(x, 5, dims=0)  # recall the input 5 steps back
   train, val = [(x[:400], y[:400])], [(x[400:], y[400:])]
   esn = nn.Sequential(
       Recurrent(
           LeakyCell(
               input=Dense(1, 100, init=uniform(1.0)),
               recurrent=Dense(100, 100, init=spectral(uniform(1.0), 0.9)),
           )
       ).requires_grad_(False),
       nn.Linear(100, 1),
   )
   result = Chain(
       IntrinsicPlasticity(lr=1e-3),
       Ridge([1e-6, 1e-3, 1.0], washout=20, score_fn=NRMSE()),
   ).fit(esn, train, val)
   print(evaluate(esn, val, NRMSE(), washout=20))

RLS updates the readout one sample at a time and keeps its state across fits,
for streams; FORCE runs a closed loop and updates the fed-back readout at every
step, as the :doc:`FORCE example <../examples/force>` shows.

Backprop
--------

:class:`~torchdyno.learners.Backprop` trains whatever requires grad, so
freezing a part with ``requires_grad_(False)`` decides what trains. It takes a
loss, an optimizer by name or as a callable given the named parameters, a
scheduler stepped per epoch or per batch, gradient clipping, a washout, a
``callback(epoch, history, model)`` that can stop training, and, with a
``score_fn`` and ``val``, keeps the best epoch's weights.

Caching features
----------------

When a frozen part of a model is shared by many fits,
:func:`~torchdyno.learners.cache_features` runs it once and returns a loader of
its outputs:

.. code-block:: python

   from torchdyno.learners import cache_features

   states = cache_features(esn[:-1], train)
   Ridge(1e-6, washout=20).fit(nn.Sequential(esn[-1]), states)  # fits esn[-1]
