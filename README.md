# Asmnet

A small x86-64 assembly math library written in NASM, with a C test harness, plus a from-scratch micro neural network in Python.

## What's in here

**`math/`** — the core assembly library (`mathlib.asm`, `matrix.asm`, `calculus.asm`, `complex.asm`). It exposes functions following the System V AMD64 ABI:

- Integer arithmetic: add, sub, mul, div, mod, neg, pow, sqrt, factorial, gcd, lcm, min, max, abs
- Floating point (SSE): fadd, fsub, fmul, fdiv, fneg, fabs
- Matrices: matmul, matadd, matsub, transpose, det2, det3
- Calculus: numerical integration and differentiation
- Complex numbers: add, sub, mul, div, abs, conj, neg

**`test/test_math.c`** — C test harness that calls the assembly functions and checks results against expected values.

**`Makefile`** — builds the `.o` files with NASM, links them with the C test harness using GCC, and runs the tests.

**`python-testing/micro.py`** — a tiny neural network written from scratch in pure Python. It implements character-level tokenization, linear algebra primitives, layer normalization, causal multi-head attention, a SwiGLU feed-forward block, residual connections, cross-entropy loss, backpropagation, and an Adam optimizer. It trains on a small corpus and can generate text completions.

**`asm-tools/`** — scratch assembly experiments and helpers (matrix multiplication, function pulling, etc.).

**`nanonet.asm`** — placeholder for a future neural network written in assembly.

## Requirements

- nasm
- gcc
- make
- python3 (for `micro.py`)

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

```bash
python3 python-testing/micro.py
```

It trains for 80 epochs on a tiny corpus and prints loss every 20 epochs, then shows a few completions for prompts like `low `, `fast`, and `high`.

## Quick one-off assembly build

`makefile.sh` is a helper that prompts for a filename, assembles it with NASM (`elf64`), links it, runs it, and cleans up:

```bash
bash makefile.sh
```

## Notes

- All assembly functions use the System V AMD64 calling convention. Integer results come back in `rax`, floating point results in `xmm0`.
- The library is position-independent friendly and built with `-no-pie` so the C test harness can call the assembly symbols directly.
- This is a personal learning project, not production-grade code.

## License

No license specified yet.