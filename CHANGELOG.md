## 0.3.0 (2026-11-12)

### Feat

- **sequence**: add a layer that mixes neighbouring units by a convolution
- **cells**: add a leaky diagonal transition
- **matrices**: add a polar initialiser for complex entries
- **testing**: add echo_state_distance
- **benchmark**: add NARMA, Mackey-Glass and memory capacity benchmarks, and UCR and UEA files
- **objectives**: add balanced accuracy, F1, confusion matrix and memory capacity, and tasks built without subclassing
- **learners**: select ridge penalties by leave-one-out, weight classes, and cache features
- **matrices**: add the cycle initialiser, per-channel scales and spectral_radius
- add MiniRocket's kernels and the quantile bias learner
- **readout**: add MultiRocket's MPV, MIPV and LSPV pooling
- add a feedback cell for closed loops and the FORCE learner
- **sequence**: add Sum, Reverse and Diff, and let Stack feed the input to every layer
- run sequences of different lengths in one batch, padded with NaN
- **matrices**: read and update any matrix through its dense form
- **layers**: take random convolution kernels from a draw
- **cells**: integrate any vector field, with one step contract for every cell
- **cells**: let linear diagonal transitions read the input and be real
- **layers**: carry the state across calls
- **layers**: continue a stack from a previous state
- **benchmark**: pass ridge options through run_conditions
- **learners**: make RLS resumable, with forgetting and a public update
- **learners**: add training controls to backprop
- keep what fits return, with ridge's score for every candidate
- **learners**: add a learner protocol and chain learners
- **learners**: fit a readout anywhere in a model, by name
- **learners**: take ridge's selection mode from its metric
- **learners**: add a washout to backprop
- compare readout, frozen and trained conditions
- add the Lorenz forecasting benchmark
- add a benchmark runner that aggregates metrics over seeds
- add an intrinsic plasticity learner
- add a recursive least squares learner
- add a closed-form continuous-time cell
- add a coupled oscillator cell
- add a parallel scan layer for linear diagonal cells
- add a linear diagonal cell with ring and zero-order-hold transitions
- support complex dense matrices and initialisers
- add ROCKET random convolution kernels
- add max and PPV pooling that skip missing positions
- add the closed-form ridge learner
- add the backprop learner and shared evaluation
- add regression and classification tasks and metrics
- add time pooling and side-by-side concatenation
- add a stack layer that concatenates every layer's output
- add a conformance check for recurrent layers
- add Euler and leaky cells and the recurrent layer
- add structured matrices and initialisers
- add seeding and RNG state capture

### Fix

- **learners**: leave absent steps out of Backprop's loss
- follow torch's default dtype in every part
- **learners**: apply intrinsic plasticity online, unit by unit
- export each module's public names and import every module with the package
- **testing**: check that a layer keeps the sequences of a batch apart
- **cells**: reject constants out of range
- **matrices**: reject sizes, bounds and coupling pairs they cannot build
- **learners**: reject empty loaders and penalties they cannot use
- **cells**: keep ring and zero-order-hold transitions finite at their edges
- **matrices**: initialise matrices in half precision
- **layers**: slice a Concat into a Concat
- **learners**: step plateau schedulers on a score
- **layers**: draw only random kernels that fit the series
- **cells**: reject matrices whose widths do not match the state
- **layers**: reject inputs that are not time-first sequences
- **reproducibility**: seed loaders' own generators
- **matrices**: draw coupling pairs from torch's generator
- **matrices**: reject spectral rescales that cannot reach the radius
- **matrices**: keep complex weights intact under dtype casts
- **cells**: keep tensor leaks and steps as buffers
- **learners**: solve on the device of the features
- **learners**: validate ridge's penalties and selection mode
- **learners**: evaluate sequences of different lengths
- **benchmark**: score benchmarks after the dataset's washout
- **learners**: score predictions after the washout
- **learners**: fit and score in eval mode, and keep the model's modes
- **learners**: reject features that are not finite
- **learners**: reject targets that do not match the readout
- **metrics**: normalise NRMSE by each output's variance
- **metrics**: require predictions and targets of the same shape

### Refactor

- **sequence**: take the input matrix first in LinearDiagonalCell, and call every state h
- organise the package by the four levels
