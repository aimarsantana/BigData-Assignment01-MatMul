# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

**Big Data · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027**
Author: Aimar Daniel Alejandro Santana

Research question: *How does matrix size n affect the runtime and peak memory of the same
basic triple-loop (i-j-k) dense matrix multiplication implemented in Python, Java and C,
under fixed and documented execution conditions?*

The report is `report/Assignment01_AlejandroSantana_Report.pdf`.

## Repository layout

```
src/c/matmul.c              C implementation (contiguous row-major block + row pointers)
src/java/MatMul.java        Java implementation (double[][])
src/python/matmul.py        Python implementation (list of lists of float, no NumPy in kernel)
scripts/generate_inputs.py  shared input matrices (binary float64, fixed seeds) + SHA-256 manifest
scripts/run_benchmarks.py   benchmark harness (warm-up, repetitions, budget, peak RSS)
scripts/analyze.py          summary statistics, scaling fit, figures, LaTeX tables
scripts/collect_environment.sh  hardware / software description
tests/validate.py           correctness tests (exact cases + NumPy reference with error bound)
config/benchmark_config.json            main study configuration
config/supplementary_alignment_config.json  separate power-of-two experiment (C, Java)
results/                    raw measurements (CSV), logs, summaries, environment
figures/                    generated plots (PDF + PNG)
report/                     LaTeX source of the report + generated tables
run_all.sh                  one command that reproduces everything
```

## Requirements (versions used in the study)

| Tool | Version |
|------|---------|
| OS | Ubuntu 24.04.5 LTS (x86-64, Linux kernel 6.18) |
| C compiler | gcc 13.3.0 |
| Java | OpenJDK 21.0.12 (HotSpot 64-Bit Server VM) |
| Python | CPython 3.13.16 |
| Analysis | numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2 |
| Report | pdflatex (TeX Live) — optional |

NumPy is used **only** to generate the inputs and as the correctness reference; it is never
used inside the benchmarked Python kernel. The peak-memory metric reads `/proc/self/status`,
so it is Linux-specific (it prints `-1` elsewhere).

```bash
pip install numpy pandas matplotlib      # add --break-system-packages on Debian/Ubuntu if needed
```

## Reproduce everything

```bash
./run_all.sh            # build -> inputs -> validation -> benchmarks -> analysis -> report
```

or step by step:

```bash
./build.sh                                   # gcc -O2 -std=c11 -Wall -Wextra ; javac --release 21
python3 scripts/generate_inputs.py           # data/inputs/A_n*.bin, B_n*.bin (+ manifest.json)
python3 tests/validate.py > results/validation.csv
./scripts/collect_environment.sh             # results/environment.txt
python3 scripts/run_benchmarks.py            # results/raw_samples.csv, runs.csv, info.csv, logs/
python3 scripts/run_benchmarks.py --config config/supplementary_alignment_config.json \
        --results results/supplementary --no-baseline
python3 scripts/analyze.py                   # results/summary.csv, figures/, report/generated/
python3 scripts/analyze_supplementary.py
cd report && latexmk -pdf Assignment01_AlejandroSantana_Report.tex
```

Running a single implementation by hand:

```bash
./build/matmul                     data/inputs/A_n512.bin data/inputs/B_n512.bin 512 1 10 1
java -Xms64m -Xmx4g -XX:+UseSerialGC -cp build MatMul data/inputs/A_n512.bin data/inputs/B_n512.bin 512 3 10 1
python3 src/python/matmul.py       data/inputs/A_n512.bin data/inputs/B_n512.bin 512 1 10 1
#   arguments: A.bin B.bin n warmup_runs measured_reps batch [C_out.bin]
```

Each program prints `SAMPLE,<lang>,<n>,<phase>,<index>,<batch>,<ns_per_multiplication>,<checksum>`
for every warm-up and measured run, and `MEM,<lang>,<n>,vmhwm_kib,<peak RSS>` at the end.

## Inputs

* Generator: NumPy `default_rng` (PCG64); seed A = 20261012, seed B = 20261013.
* Values: uniform in [-1, 1), float64, stored as raw little-endian row-major binary.
* All three languages read the **same files**, so the logical inputs are identical.
  `data/inputs/manifest.json` stores the SHA-256 of every file. The input files are not
  committed (≈133 MB); regenerate them with `scripts/generate_inputs.py`.

## Experimental protocol (summary — details in the report)

* Timer boundary: **multiplication kernel only** (input loading and output allocation are outside).
  Monotonic timers: `clock_gettime(CLOCK_MONOTONIC)`, `System.nanoTime()`, `time.perf_counter_ns()`.
* Sizes: 64, 128, 256, 384, 512, 768, 1024, 1536, 2048.
* Repetitions: 10 measured (n ≤ 512) / 5 measured (n > 512), one fresh process per (language, n).
  Warm-up runs (excluded, but kept in the raw data): C 1, Java 3, Python 1.
  For n = 64 and 128 each sample times a batch of 20 / 5 multiplications and reports time per multiplication.
* Budget: 200 s per single multiplication; 4 GB peak RSS. A language that exceeds the budget
  is not attempted at larger sizes (recorded as `TIMEOUT` / `NOT_ATTEMPTED`).
* Memory metric: peak resident set size `VmHWM` read by each process from `/proc/self/status`.
* Statistics: median and interquartile range of the measured samples.
* Every row of the raw CSV files is labelled with language (method), n, numeric type (float64), thread count (1), phase and repetition index.

## AI-use disclosure

Generative AI (Claude, Anthropic) was used as described in the "AI-use disclosure" section of
the report. All measurements in `results/` were produced by running the code in this repository.
