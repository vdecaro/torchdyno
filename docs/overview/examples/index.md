# Examples

Each example composes a well-known model from TorchDyno's parts, fits it and
prints its score, in seconds. Most run on a small synthetic task: noisy sine
waves classified by their frequency, built by `examples/data.py`.

The code for the examples is available in the
:fontawesome-brands-github:
[examples](https://github.com/vdecaro/torchdyno/tree/main/examples/)
subdirectory of the repository.

## Reservoirs

<div class="index-list" markdown>

-   [Echo state network](esn.md)

    A frozen leaky reservoir with a sparse recurrent matrix forecasts the Lorenz system, read out by ridge regression.

-   [Deep echo state network](deep_esn.md)

    Three reservoirs run in series on the Lorenz system, read out over every layer's states.

-   [ParalESN](paralesn.md)

    Deep reservoir layers of linear complex modes, run by a parallel scan and mixed across modes, forecast the Lorenz system.

-   [Intrinsic plasticity](esn_plasticity.md)

    Intrinsic plasticity adapts a saturated reservoir before its ridge readout.

</div>

## Online learning

<div class="index-list" markdown>

-   [Online readout](rls_forcing.md)

    Recursive least squares fits a reservoir's readout online, then the reservoir forecasts in closed loop.

-   [FORCE](force.md)

    A reservoir with no input learns by FORCE to generate a two-tone wave.

</div>

## Trained recurrent networks

<div class="index-list" markdown>

-   [AdaDiag](adadiag.md)

    A sparse assembly of diagonal modules with an adaptive step, trained end to end.

-   [SCN](scn.md)

    Fixed block-diagonal modules coupled by a trained skew-symmetric matrix: the blocks are frozen, while the coupling, the input and the readout train.

-   [coRNN](cornn.md)

    Coupled damped oscillators, trained end to end.

-   [CfC](cfc.md)

    A closed-form continuous-time cell with a small backbone, trained end to end.

</div>

## State space models

<div class="index-list" markdown>

-   [LRU](lru.md)

    A linear complex diagonal recurrence run with a parallel scan, trained end to end.

-   [S4D](s4d.md)

    S4D as a frozen feature extractor, pooled like ROCKET and read out by ridge.

</div>

## Random convolutions

<div class="index-list" markdown>

-   [ROCKET](rocket.md)

    Ten thousand random convolution kernels, PPV and max pooling, and a ridge classifier on standardised features, as in the ROCKET paper.

-   [MiniRocket](minirocket.md)

    MiniRocket's fixed kernel patterns, their biases set from quantiles of the kernels' outputs on the training series, PPV pooling and a ridge classifier.

</div>

## Comparisons

<div class="index-list" markdown>

-   [Readout, frozen, trained](conditions.md)

    One AdaDiag model run three ways on the same seeds: only the readout fitted by ridge; the same parts frozen, with a linear encoder in front and the readout trained by backprop; and everything trained.

</div>
