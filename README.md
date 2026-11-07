# Liquid: a Liquid Time-Constant network in Mojo

This is a small Liquid Time-Constant (LTC) neural network written in Mojo, with no PyTorch or TensorFlow. I wrote it to learn how continuous-time networks work and to see how fast I could make one run on a CPU.

The LTC model comes from Hasani et al., "Liquid Time-constant Networks" (AAAI 2021, arXiv:2006.04439). This is my own implementation, not an official one.

## What it does

The network's hidden state changes according to an ODE:

```
dx/dt = -(1/tau + f) * x + f * A
```

Here `f` depends on the input and the current state, `tau` is a time constant, and `A` is a bias term. Each timestep, the code integrates this ODE forward instead of applying a plain layer.

## How it's built

- **Solver (`src/liquid_ode.mojo`).** An RK4 integrator with an adaptive step size. It uses step-doubling: take one full step, then two half steps, and compare the results to estimate the error. I chose this over Dormand-Prince because it was simpler to get right.
- **State layout (`src/liquid_state.mojo`).** Hidden activations, tau, and step-size data are stored as separate arrays (struct of arrays) so SIMD loads are contiguous. The arrays are split into tiles of 16 float32 values, which is 64 bytes, one cache line.
- **Network (`src/liquid_network.mojo`).** A cell is one layer, and a network is a stack of cells. Work is split across hidden-unit tiles and run in parallel. The state is double-buffered: each step reads from one buffer and writes to the other, so threads never read a half-updated value. Timesteps themselves still run one after another, since each depends on the last.
- **Demo (`src/main.mojo`).** Builds a 128-unit network, feeds it a synthetic sensor signal, and prints timing and the final state.

The inference loop does no dynamic allocation.

## Build and run

```
cd src
mojo build main.mojo -o liquid_demo -O3
./liquid_demo
```

Tested on: [Mojo version, OS, CPU].

## Status

What works: it compiles and the demo runs.

What I have not tested:
- accuracy against a reference LTC implementation
- any real robot or embedded hardware

Numbers from my machine: about 30 microseconds per timestep for 128 units on CPU. These are rough and not a proper benchmark.

## Mojo version notes

Mojo's syntax has changed a lot, and older tutorials will not compile. This code uses [version]. The two places I had to adjust:

- `UnsafePointer` origin parameter.
- The `parallelize` call in `src/liquid_network.mojo`.

## Credits

Made by kds1123001.

Licensed under MIT (see `LICENSE`).
MEEE cuz u know i made it muheheheheheh
+no llm was used+
