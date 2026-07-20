# Liquid — a Mojo-native Liquid Time-Constant Network for edge robotics

A from-scratch Liquid Neural Network (LTC-style, MIT CSAIL formulation) with:

1. a SIMD-vectorized, adaptive-step RK4 ODE solver (`src/liquid_ode.mojo`)
2. a Struct-of-Arrays, cache-tiled state/weight layout (`src/liquid_state.mojo`)
3. a lock-free parallel timestep executor using double-buffered state (`src/liquid_network.mojo`)

No PyTorch/TensorFlow. No dynamic allocation inside the inference loop.

## A note on Mojo's API churn — read this before you build

Mojo is moving fast; as of the 1.0 beta line (mid-2026) the language dropped
`fn`/`let`/`inout`/`borrowed`/`owned`/`alias`/`@parameter` as the *default*
idiom in favor of:

| Old (pre-1.0)            | Current (1.0 beta)              |
|---------------------------|----------------------------------|
| `fn` everywhere            | `def` for new code (`fn` still compiles, legacy) |
| `let x = ...`              | removed — everything is `var`   |
| `inout self`               | `mut self`                      |
| `borrowed self` (default)  | plain `self` (read is default)  |
| `owned x`                  | `var x`                         |
| `alias` / `@parameter`     | `comptime`                      |
| `DTypePointer[T]`          | `UnsafePointer[T, origin=...]` (origin now explicit, no safe default) |

This code is written against the **current** conventions. Because I don't have
a Mojo toolchain available in this environment to compile against (network
access here is restricted to package registries, not `modular.com`), I
haven't been able to run `mojo build` on this myself — treat it as a careful,
idiomatic first draft rather than a compiled artifact. The two spots most
likely to need a one-line adjustment for your exact nightly are:

- `UnsafePointer`'s origin parameter name/type (`MutUntrackedOrigin` vs
  `MutExternalOrigin` — both names appear in current docs depending on
  nightly; I used `MutUntrackedOrigin`, swap it if your compiler rejects it).
- `parallelize`'s exact closure-capture signature in `src/liquid_network.mojo`
  — the `algorithm.parallelize` call shape has changed a few times. I've
  isolated it into a single function (`_run_parallel_tiles`) so if it doesn't
  match your stdlib, it's a one-function fix.

If you hit a compile error, paste it back to me and I'll patch it — that's a
much tighter loop than me guessing at a moving target from here.

## Files

- `src/liquid_state.mojo` — `LiquidState`: SoA storage for hidden activations,
  time-constants (tau), and per-unit adaptive step-size/error, tiled into
  cache-line-sized (16×float32 = 64B) chunks. Owns its memory, frees on `deinit`.
- `src/liquid_ode.mojo` — the LTC dynamics `dx/dt = -(1/tau + f)·x + f·A` and
  a SIMD-width adaptive RK4 integrator using step-doubling (Richardson
  extrapolation) for error control — cheaper to get right than a full
  Fehlberg tableau and still gives genuine adaptive depth.
- `src/liquid_network.mojo` — `LiquidCell` (one layer) and `LiquidNetwork`
  (stacked layers), parallelized across hidden-unit tiles with double
  buffering so tiles need zero synchronization within a timestep.
- `src/main.mojo` — builds a 128-unit network, feeds it a synthetic sensor
  stream, runs it, prints timing and final state.

## Build

```
cd src
mojo build main.mojo -o liquid_demo -O3
./liquid_demo
```
## credits  
MEEE cuz u know i made it muheheheheheh
