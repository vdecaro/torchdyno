===============
Getting started
===============

.. toctree::
   :maxdepth: 2
   :hidden:

   first-reservoir
   train-end-to-end
   benchmark-across-seeds

Let's get started with TorchDyno.

Each guide below builds a complete model and runs in seconds. Once you are
comfortable with the workflow, move to the :doc:`concept pages
<../concepts/index>` to see every part each level offers.

Happy paths
-----------

.. grid:: 1

   .. grid-item::

      .. card:: 1. Read out a reservoir
         :link: first-reservoir
         :link-type: doc

         Compose an echo state network, fit its readout in closed form, and
         score its forecasts.

   .. grid-item::

      .. card:: 2. Train a model end to end
         :link: train-end-to-end
         :link-type: doc

         Compose a classifier of sequences from a different cell and train
         every part of it by backprop.

   .. grid-item::

      .. card:: 3. Benchmark across seeds
         :link: benchmark-across-seeds
         :link-type: doc

         Fit fresh models on several seeds and report every metric as a mean
         and a standard deviation.

.. tip::

   If you have not yet installed TorchDyno, refer to the :doc:`Installation <../install>` page.
