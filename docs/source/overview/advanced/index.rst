========
Advanced
========

.. toctree::
   :maxdepth: 2
   :hidden:

   variable-length
   custom-parts
   benchmarks

These pages cover what you need once the basic workflow is familiar.

.. grid:: 1

   .. grid-item::

      .. card:: Sequences of different lengths
         :link: variable-length
         :link-type: doc

         Pad shorter sequences with NaN and let every part skip the steps they
         do not have.

   .. grid-item::

      .. card:: Writing your own parts
         :link: custom-parts
         :link-type: doc

         Add a matrix, a cell, a kernel draw or a learner, and check it with
         the conformance kit.

   .. grid-item::

      .. card:: Benchmarks
         :link: benchmarks
         :link-type: doc

         Run models across seeds on built-in or UCR and UEA datasets, compare
         readout-only, frozen and trained models, and reproduce a run.
