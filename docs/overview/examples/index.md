# Examples

Each example composes a well-known model from TorchDyno's parts, fits it and
prints its score, in seconds. Most run on a small synthetic task: noisy sine
waves classified by their frequency, built by `examples/data.py`.

The code for the examples is available in the
:fontawesome-brands-github:
[examples](https://github.com/vdecaro/torchdyno/tree/main/examples/)
subdirectory of the repository.

<div class="grid cards" markdown>

-   :octicons-pulse-16: **[Echo state network](esn.md)**

    A frozen leaky reservoir with a sparse recurrent matrix forecasts the Lorenz system, read out by ridge regression.

-   :octicons-stack-16: **[Deep echo state network](deep_esn.md)**

    Three reservoirs run in series on the Lorenz system, read out over every layer's states.

-   :octicons-sliders-16: **[Intrinsic plasticity](esn_plasticity.md)**

    Intrinsic plasticity adapts a saturated reservoir before its ridge
    readout.

-   :octicons-sync-16: **[Online readout](rls_forcing.md)**

    Recursive least squares fits a reservoir's readout online, then the
    reservoir forecasts in closed loop.

-   :octicons-iterations-16: **[FORCE](force.md)**

    A reservoir with no input learns by FORCE to generate a two-tone wave.

-   :octicons-cpu-16: **[AdaDiag](adadiag.md)**

    A sparse assembly of diagonal modules with an adaptive step, trained end to end.

-   :octicons-lock-16: **[SCN](scn.md)**

    Fixed block-diagonal modules coupled by a trained skew-symmetric matrix: the blocks are frozen, while the coupling, the input and the readout train.

-   :octicons-goal-16: **[coRNN](cornn.md)**

    Coupled damped oscillators, trained end to end.

-   :octicons-clock-16: **[CfC](cfc.md)**

    A closed-form continuous-time cell with a small backbone, trained end to end.

-   :octicons-zap-16: **[LRU](lru.md)**

    A linear complex diagonal recurrence run with a parallel scan, trained end to end.

-   :octicons-graph-16: **[S4D](s4d.md)**

    S4D as a frozen feature extractor, pooled like ROCKET and read out by ridge.

-   :octicons-rocket-16: **[ROCKET](rocket.md)**

    Ten thousand random convolution kernels, PPV and max pooling, and a ridge classifier on standardised features, as in the ROCKET paper.

-   :octicons-rocket-16: **[MiniRocket](minirocket.md)**

    MiniRocket's fixed kernel patterns, their biases set from quantiles of the kernels' outputs on the training series, PPV pooling and a ridge classifier.

-   :octicons-versions-16: **[Readout, frozen, trained](conditions.md)**

    One AdaDiag model run three ways on the same seeds: only the readout fitted by ridge; the same parts frozen, with a linear encoder in front and the readout trained by backprop; and everything trained.

</div>
