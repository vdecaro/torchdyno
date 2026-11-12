coRNN
=====

Coupled damped oscillators, trained end to end. The model is an :class:`~torchdyno.sequence.OscillatorCell` run by :class:`~torchdyno.sequence.Recurrent`, pooled by :class:`~torchdyno.readout.Last` and read out by an ``nn.Linear``, trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- :class:`~torchdyno.sequence.OscillatorCell`, whose state holds positions and velocities side by side,
- why the recurrent matrix reads ``2 * H`` features while the state has ``H`` units per half,
- how the damping and the stiffness are set as cell parameters.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.cornn

It prints the accuracy on the test batches:

.. code-block:: text

   coRNN test accuracy: 1.0000

The last digits can differ across machines and PyTorch versions.

The data
--------

``frequency_task`` from ``examples/data.py`` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches ``(60, B, 1)``.

.. literalinclude:: ../../../../examples/data.py
   :language: python
   :pyobject: frequency_task

The model
---------

The cell integrates a second-order system of coupled damped oscillators with a step ``dt`` of 0.05. Its state is the positions and the velocities of ``H = 64`` oscillators, concatenated, so it has ``2 * H`` features. The recurrent matrix therefore maps ``2 * H`` features to ``H`` positions, and the input matrix maps the single input feature to ``H``. Both are :class:`~torchdyno.matrices.Dense` maps, level 1: the input bias starts at zero and the recurrent weights come from ``uniform(H**-0.5)``. ``gamma`` is the restoring force and ``epsilon`` the damping; each can also be a per-unit tensor. :class:`~torchdyno.readout.Last`, level 3, keeps the final state, ``(B, 2 * H)``, and the readout maps it to the three classes.

.. literalinclude:: ../../../../examples/cornn.py
   :language: python
   :start-at: seed_all(0)
   :end-before: Backprop(

Training and scoring
--------------------

:class:`~torchdyno.learners.Backprop` trains every parameter with Adam, at a learning rate of ``1e-2``, for 30 epochs on the cross-entropy loss. :func:`~torchdyno.learners.evaluate` scores the test batches with :class:`~torchdyno.objectives.Accuracy`.

.. literalinclude:: ../../../../examples/cornn.py
   :language: python
   :start-at: Backprop(

What to try
-----------

- Change ``dt`` to 0.1 or 0.01 to see how the integration step trades speed for stability.
- Change ``gamma`` and ``epsilon``, and see how the restoring force and the damping affect the accuracy.
- Pool with :class:`~torchdyno.readout.Mean` instead of :class:`~torchdyno.readout.Last`.
- Raise ``H`` to 128, and change the readout to ``nn.Linear(2 * H, 3)`` accordingly.

.. dropdown:: Full code

   .. literalinclude:: ../../../../examples/cornn.py
      :language: python

Related concepts
----------------

- :doc:`../concepts/sequence`
- :doc:`../concepts/readout`
- :doc:`../getting-started/train-end-to-end`
