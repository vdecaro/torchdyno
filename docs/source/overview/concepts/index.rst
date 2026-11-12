========
Concepts
========

.. toctree::
   :maxdepth: 2
   :hidden:

   matrices
   sequence
   readout
   objectives
   learners

A TorchDyno model is a plain ``torch.nn.Sequential``, and each of its parts
belongs to one of four levels:

1. :mod:`torchdyno.matrices`: how a weight matrix is structured, initialised
   and applied;
2. :mod:`torchdyno.sequence`: how a ``(T, B, F)`` sequence becomes a
   ``(T, B, H)`` one, by a recurrent cell run over time or by random
   convolutions;
3. :mod:`torchdyno.readout`: pooling over time to ``(B, H)``, then an
   ``nn.Linear``;
4. :mod:`torchdyno.objectives`: the loss and metrics a model is trained and
   scored for.

:mod:`torchdyno.learners` fits a model, by closed-form, online, local or
gradient-based rules. A part never assumes which learner will fit it, so the
same model can be read out by ridge regression or trained by backprop.

The library is a toolkit of parts, not a collection of named models. An echo
state network, a deep reservoir, a ROCKET classifier and an S4D feature
extractor are all compositions of the same few parts, and they appear as such
in the :doc:`examples <../examples/index>`, never as classes of the package.

How the levels fit
------------------

Data flows through a model in one direction, and each level hands the next a
fixed shape.

.. code-block:: python

   import torch
   from torch import nn

   from torchdyno.learners import Ridge, evaluate
   from torchdyno.matrices import Dense, spectral, uniform
   from torchdyno.objectives import Accuracy
   from torchdyno.readout import Last
   from torchdyno.sequence import LeakyCell, Recurrent

   model = nn.Sequential(
       Recurrent(  # sequence: (T, B, F) to (T, B, H), here H = 64
           LeakyCell(
               input=Dense(2, 64, init=uniform(1.0)),  # matrices
               recurrent=Dense(64, 64, init=spectral(uniform(1.0), 0.9)),
               leak=0.5,
           )
       ).requires_grad_(False),
       Last(),  # readout: pool over time to (B, H)
       nn.Linear(64, 3),  # readout: (B, H) to (B, O)
   )
   labels = torch.randint(3, (100,))
   x = torch.randn(30, 100, 2) + labels[None, :, None]
   Ridge(1e-3).fit(model, [(x, labels)])  # learner: fits the last module
   print(evaluate(model, [(x, labels)], Accuracy()))  # objective: scores it

Four conventions run through every level and are worth knowing before the
details.

- **Time first.** Sequences are ``(T, B, F)``: ``T`` timesteps of ``B``
  sequences with ``F`` features, and every loader yields ``(x, y)`` batches of
  that layout.
- **NaN marks a step a sequence does not have.** Sequences of different lengths
  share a batch padded with NaN. Recurrent layers hold their state over such a
  step, pooling skips it and the learners drop those rows. See
  :doc:`../advanced/variable-length`.
- **Freezing is ``requires_grad_(False)``.** It never changes a part's
  initialisation, so a frozen reservoir is the random reservoir it was drawn
  as. Which parts are frozen decides what a gradient-based learner trains.
- **The state is the output, and the cell is what you compile.** A cell returns
  its new state, which a layer collects, so the readout reads states directly.
  To speed a recurrence up, compile the cell with ``layer.cell.compile()``, not
  the model.

Where each part goes
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Level
     - Typical parts
     - Read next
   * - Matrices
     - ``Dense``, ``Diagonal``, ``BlockDiagonal``, ``SkewCoupling``, and the
       initialisers ``uniform``, ``normal``, ``cycle`` and ``spectral``
     - :doc:`matrices`
   * - Sequence maps
     - ``LeakyCell`` and the other cells, ``Recurrent`` and ``Scan``,
       ``RandomConv``, ``Stack``, ``Concat``, ``Sum``
     - :doc:`sequence`
   * - Readout
     - ``Last``, ``Mean``, ``Max``, ``PPV`` and kin, then ``nn.Linear``
     - :doc:`readout`
   * - Objectives
     - ``Accuracy``, ``NRMSE`` and other metrics, and the tasks that bundle
       them with a loss
     - :doc:`objectives`
   * - Learners
     - ``Ridge``, ``RLS``, ``FORCE``, ``Backprop``, ``IntrinsicPlasticity``,
       ``QuantileBias``, ``Chain``
     - :doc:`learners`

Every level is open: a part only has to follow its level's contract, and the
rest of the library works with it. :doc:`../advanced/custom-parts` shows how to
write one. To see the parts at work first, follow
:doc:`../getting-started/first-reservoir`.

.. grid:: 2

   .. grid-item::

      .. card:: Matrices
         :link: matrices
         :link-type: doc

         Structured matrices, their initialisers and bounds, and reading and
         updating any matrix through its dense form.

   .. grid-item::

      .. card:: Sequence maps
         :link: sequence
         :link-type: doc

         Cells, the layers that run them over time, random convolutions, and
         the ways to combine them.

   .. grid-item::

      .. card:: Readout
         :link: readout
         :link-type: doc

         Pooling a sequence over time, and the linear map to outputs.

   .. grid-item::

      .. card:: Objectives
         :link: objectives
         :link-type: doc

         Metrics, tasks, and scoring a model over a loader.

   .. grid-item::

      .. card:: Learners
         :link: learners
         :link-type: doc

         Ridge regression, recursive least squares, FORCE, backprop,
         intrinsic plasticity and quantile biases, alone or chained.
