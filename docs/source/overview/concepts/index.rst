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
