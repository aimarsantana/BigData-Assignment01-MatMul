#!/usr/bin/env bash
# Reproduce the whole study: build, inputs, validation, benchmarks, analysis, report.
# Total runtime on the reference machine: about 1.5 hours (dominated by Python at n >= 768).
set -euo pipefail
cd "$(dirname "$0")"

./build.sh
python3 scripts/generate_inputs.py
python3 scripts/generate_inputs.py --sizes 500 511 513 1000 1023 1025
mkdir -p results/supplementary
python3 tests/validate.py > results/validation.csv
./scripts/collect_environment.sh
python3 scripts/run_benchmarks.py | tee results/benchmark_progress.log
python3 scripts/run_benchmarks.py --config config/supplementary_alignment_config.json \
        --results results/supplementary --no-baseline | tee results/supplementary/progress.log
python3 scripts/analyze.py
python3 scripts/analyze_supplementary.py
if command -v latexmk >/dev/null; then
  (cd report && latexmk -pdf -interaction=nonstopmode Assignment01_AlejandroSantana_Report.tex)
fi
