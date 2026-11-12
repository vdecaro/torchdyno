# Intrinsic plasticity

Intrinsic plasticity adapts a saturated reservoir before its ridge readout. A strong input drives about half of the reservoir's units into saturation; the same reservoir is read out twice, as initialised and after [`IntrinsicPlasticity`][torchdyno.learners.IntrinsicPlasticity] has adapted its gains and biases toward unsaturated outputs, on the Lorenz forecasting task.

Use it when you want to understand:

- how [`IntrinsicPlasticity`][torchdyno.learners.IntrinsicPlasticity] adapts a reservoir without a target,
- how to compare a model with and without a learner by copying it,
- how adapting the reservoir can lower the error of the same ridge readout.

Run it from the repository root with:

```bash
uv run python -m examples.esn_plasticity
```

It prints the NRMSE on the evaluation segment for each reservoir:

```text
without intrinsic plasticity: test NRMSE 2.13e-04
with intrinsic plasticity: test NRMSE 7.19e-05
```

The last digits can differ across machines and PyTorch versions.

## The data and the model

The data are the Lorenz benchmark of the [echo state network](esn.md) example. The reservoir is built to saturate: the input weights come from `uniform(6.0)` and the bias from `uniform(3.0)`, large against the `tanh` activation of the [`LeakyCell`][torchdyno.sequence.LeakyCell]. The recurrent matrix has a spectral radius of 0.95. The reservoir is frozen with `requires_grad_(False)`, and an `nn.Linear` readout follows.

```python
--8<-- "examples/esn_plasticity.py:19:30"
```

## Adapting a copy

`copy.deepcopy` keeps the original reservoir for the comparison. [`IntrinsicPlasticity`][torchdyno.learners.IntrinsicPlasticity] runs the training loader through the reservoir and nudges each unit's gain and bias, with step size `lr`, so that its output approaches a Gaussian of mean `mu` (0 by default) and standard deviation `sigma`. It ignores targets and changes only the gains and biases, even though the reservoir is frozen to backprop.

```python
--8<-- "examples/esn_plasticity.py:31:32"
```

## Fitting both

Each model then gets the same ridge readout, fitted on the same loader with the same penalty, and is scored on the evaluation loader. Any difference comes from the reservoir.

```python
--8<-- "examples/esn_plasticity.py:34:37"
```

## What to try

- Chain the two learners instead of copying: `Chain(IntrinsicPlasticity(lr=1e-3, sigma=0.2), Ridge(l2=1e-6, washout=data.washout))`, with [`Chain`][torchdyno.learners.Chain].
- Increase `epochs` of [`IntrinsicPlasticity`][torchdyno.learners.IntrinsicPlasticity], or change `sigma`, and watch the error.
- Lower the input scale from `uniform(6.0)` to `uniform(1.0)` and see how much plasticity still helps on an unsaturated reservoir.

??? example "Full code"

    ```python
    --8<-- "examples/esn_plasticity.py"
    ```

## Related concepts

- [Read out a reservoir](../getting-started/first-reservoir.md)
- [Learners](../concepts/learners.md)
- [Sequence maps](../concepts/sequence.md)
