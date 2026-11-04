# Asmnet

A math library written in x86 NASM, with C based harness

## What's in here

**`math/`** — the core assembly library (`mathlib.asm`, `matrix.asm`, `calculus.asm`, `complex.asm`, `vector.asm`). It exposes functions following the System V AMD64 ABI:

- Integer arithmetic: add, sub, mul, div, mod, neg, pow, sqrt, factorial, gcd, lcm, min, max, abs
- Floating point (SSE): fadd, fsub, fmul, fdiv, fneg, fabs
- Scalar math: sqrt, sin, cos, exp (e^x), log (ln), square, sigmoid, erf, threshold
- Matrices: matmul, matadd, matsub, transpose, det2, det3
- Vectors: lookup, dot, row-sum, cumsum, mean, variance, rms, element-wise add/sub/mul/div, scalar mul/div/add/sub, sort, threshold
- Calculus: numerical integration and differentiation
- Complex numbers: add, sub, mul, div, abs, conj, neg

**`test/test_math.c`** — C test harness that calls the assembly functions and checks results against expected values.

**`Makefile`** — builds the `.o` files with NASM, links them with the C test harness using GCC, and runs the tests.

**`example_projects/micro_nn.py`** — a from-scratch micro neural network in Python that calls the assembly math functions (via ctypes / `build/libasmnet.so`) for its linear algebra. It implements character-level tokenization, linear algebra primitives, layer normalization, causal multi-head attention, a SwiGLU feed-forward block, residual connections, cross-entropy loss, backpropagation, and an Adam optimizer. It trains on a small corpus and can generate text completions.

**`asm-tools/`** — scratch assembly experiments and helpers (matrix multiplication, function pulling, etc.).

**`nanonet.asm`** — placeholder for a future neural network written in assembly.

## Requirements

- nasm
- gcc
- make
- python3 (for `micro_nn.py`)

## Building and running the tests

```bash
make
make test
```

Or build and run in one step:

```bash
make run
```

Clean up build artifacts:

```bash
make clean
```

The Makefile compiles each `.asm` file in `math/` into an object file, compiles `test/test_math.c`, links everything together, and runs the resulting binary. It prints a PASS/FAIL line per test and exits non-zero if any test fails.

## Running the Python micro network

The example project loads `build/libasmnet.so` (built with `make lib`) and uses the
assembly `matmul`, `matadd`, `transpose`, and SSE scalar ops for its linear algebra:

```bash
make lib
python3 example_projects/micro_nn.py
```

It trains for 80 epochs on a tiny corpus and prints loss every 20 epochs, then shows a few completions for prompts like `low `, `fast`, and `high`.

Build the shared library separately:

```bash
make lib
```

## Quick one-off assembly build

`makefile.sh` is a helper that prompts for a filename, assembles it with NASM (`elf64`), links it, runs it, and cleans up:

```bash
bash makefile.sh
```

## Notes

- All assembly functions use the System V AMD64 calling convention. Integer results come back in `rax`, floating point results in `xmm0`.
- The library is position-independent friendly and built with `-no-pie` so the C test harness can call the assembly symbols directly.
