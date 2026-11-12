========
Examples
========

.. toctree::
   :maxdepth: 2
   :hidden:

   esn
   deep_esn
   esn_plasticity
   rls_forcing
   force
   adadiag
   scn
   cornn
   cfc
   lru
   s4d
   rocket
   minirocket
   conditions

Each example composes a well-known model from TorchDyno's parts, fits it and
prints its score, in seconds. Most run on a small synthetic task: noisy sine
waves classified by their frequency, built by ``examples/data.py``.

The code for the examples is available in the
:icon:`fa-brands fa-github`
`examples <https://github.com/vdecaro/torchdyno/tree/main/examples/>`_
subdirectory of the repository.

.. grid:: 1

   .. grid-item::

      .. card:: :octicon:`pulse;1em;info` **Echo state network**
         :link-type: doc
         :link: esn

         A frozen leaky reservoir with a sparse recurrent matrix forecasts the Lorenz system, read out by ridge regression.

   .. grid-item::

      .. card:: :octicon:`stack;1em;info` **Deep echo state network**
         :link-type: doc
         :link: deep_esn

         Three reservoirs run in series on the Lorenz system, read out over every layer's states.

   .. grid-item::

      .. card:: :octicon:`sliders;1em;info` **Intrinsic plasticity**
         :link-type: doc
         :link: esn_plasticity

         Intrinsic plasticity adapts a saturated reservoir before its ridge
         readout.

   .. grid-item::

      .. card:: :octicon:`sync;1em;info` **Online readout**
         :link-type: doc
         :link: rls_forcing

         Recursive least squares fits a reservoir's readout online, then the
         reservoir forecasts in closed loop.

   .. grid-item::

      .. card:: :octicon:`iterations;1em;info` **FORCE**
         :link-type: doc
         :link: force

         A reservoir with no input learns by FORCE to generate a two-tone wave.

   .. grid-item::

      .. card:: :octicon:`cpu;1em;info` **AdaDiag**
         :link-type: doc
         :link: adadiag

         A sparse assembly of diagonal modules with an adaptive step, trained end to end.

   .. grid-item::

      .. card:: :octicon:`lock;1em;info` **SCN**
         :link-type: doc
         :link: scn

         Fixed block-diagonal modules coupled by a trained skew-symmetric matrix: the blocks are frozen, while the coupling, the input and the readout train.

   .. grid-item::

      .. card:: :octicon:`goal;1em;info` **coRNN**
         :link-type: doc
         :link: cornn

         Coupled damped oscillators, trained end to end.

   .. grid-item::

      .. card:: :octicon:`clock;1em;info` **CfC**
         :link-type: doc
         :link: cfc

         A closed-form continuous-time cell with a small backbone, trained end to end.

   .. grid-item::

      .. card:: :octicon:`zap;1em;info` **LRU**
         :link-type: doc
         :link: lru

         A linear complex diagonal recurrence run with a parallel scan, trained end to end.

   .. grid-item::

      .. card:: :octicon:`graph;1em;info` **S4D**
         :link-type: doc
         :link: s4d

         S4D as a frozen feature extractor, pooled like ROCKET and read out by ridge.

   .. grid-item::

      .. card:: :octicon:`rocket;1em;info` **ROCKET**
         :link-type: doc
         :link: rocket

         Ten thousand random convolution kernels, PPV and max pooling, and a ridge classifier on standardised features, as in the ROCKET paper.

   .. grid-item::

      .. card:: :octicon:`rocket;1em;info` **MiniRocket**
         :link-type: doc
         :link: minirocket

         MiniRocket's fixed kernel patterns, their biases set from quantiles of the kernels' outputs on the training series, PPV pooling and a ridge classifier.

   .. grid-item::

      .. card:: :octicon:`versions;1em;info` **Readout, frozen, trained**
         :link-type: doc
         :link: conditions

         One AdaDiag model run three ways on the same seeds: only the readout fitted by ridge; the same parts frozen, with a linear encoder in front and the readout trained by backprop; and everything trained.
