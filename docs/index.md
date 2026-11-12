---
template: home.html
title: TorchDyno
hide:
  - navigation
  - toc
---

<div class="td-hero" markdown>

<div class="td-hero__text" markdown>

# Neural dynamical systems, built from parts

Compose reservoirs, state space models and ODE-based networks as plain
`nn.Sequential` models. Fit the readout in closed form, learn online, or train
everything by backprop.

<div class="td-hero__actions" markdown>

[Read out your first reservoir](overview/getting-started/first-reservoir.md){ .md-button .md-button--primary }
[Install](overview/install.md){ .md-button }

</div>

<div class="td-hero__install" markdown>

```bash
pip install torchdyno
```

</div>

</div>

<figure class="td-hero__live" markdown>
<canvas class="td-states" aria-hidden="true"></canvas>
<figcaption>A 32-unit leaky reservoir, simulated live in your browser. The top
row is the input; each row below is one unit's state over time, orange above
zero and magenta below, faint where the unit barely responds.</figcaption>
</figure>

</div>

## One model, four levels

There is no model zoo. An echo state network is a cell, two matrices and a
linear readout; every other model is another arrangement of the same parts.

<div class="td-split" markdown>

<div markdown>

```python
esn = nn.Sequential(
    Recurrent(  # 2. sequence
        LeakyCell(
            input=Dense(3, 300, init=uniform(1.0)),  # 1. matrices
            recurrent=Dense(300, 300, init=spectral(uniform(1.0), 0.9)),
            leak=0.5,
        )
    ).requires_grad_(False),
    nn.Linear(300, 3),  # 3. readout
)
Ridge(l2=[1e-8, 1e-6]).fit(esn, train, val)  # a learner
evaluate(esn, test, NRMSE())  # 4. objectives
```

</div>

<div class="index-list" markdown>

1.  [Matrices](overview/concepts/matrices.md)

    How a weight matrix is structured, initialised and applied: dense,
    diagonal, block-diagonal, skew-symmetric or complex.

2.  [Sequence](overview/concepts/sequence.md)

    How a sequence is processed: a recurrent cell run step by step or by a
    parallel scan, or random convolutions.

3.  [Readout](overview/concepts/readout.md)

    Pooling over time, then a linear map to the outputs.

4.  [Objectives](overview/concepts/objectives.md)

    The task a model is fitted for, and the metrics it is scored by.

</div>

</div>

## Fit it, or train it

<div class="td-columns" markdown>

<div markdown>

### Fitted, without gradients

Freeze the dynamics and fit the readout: ridge regression in closed form,
recursive least squares or FORCE online, with intrinsic plasticity to adapt the
reservoir first.

[Learners](overview/concepts/learners.md)

</div>

<div markdown>

### Trained end to end

Backprop trains every parameter that requires a gradient. Freeze any part with
`requires_grad_(False)` and train the rest around it.

[Train a model end to end](overview/getting-started/train-end-to-end.md)

</div>

<div markdown>

### Compared on the same seeds

Run one model three ways, readout only, frozen and fully trained, and report
each across seeds.

[Benchmark across seeds](overview/getting-started/benchmark-across-seeds.md)

</div>

</div>

## Models made from the same parts

Each example composes a known model, fits it and prints its score in seconds.

<div class="td-columns td-columns--narrow" markdown>

<div markdown>

### Reservoirs

- [Echo state network](overview/examples/esn.md)
- [Deep echo state network](overview/examples/deep_esn.md)
- [Intrinsic plasticity](overview/examples/esn_plasticity.md)

</div>

<div markdown>

### Online learning

- [Recursive least squares](overview/examples/rls_forcing.md)
- [FORCE](overview/examples/force.md)

</div>

<div markdown>

### Trained RNNs

- [AdaDiag](overview/examples/adadiag.md)
- [SCN](overview/examples/scn.md)
- [coRNN](overview/examples/cornn.md)
- [CfC](overview/examples/cfc.md)

</div>

<div markdown>

### State space models

- [LRU](overview/examples/lru.md)
- [S4D](overview/examples/s4d.md)

</div>

<div markdown>

### Random convolutions

- [ROCKET](overview/examples/rocket.md)
- [MiniRocket](overview/examples/minirocket.md)

</div>

</div>
