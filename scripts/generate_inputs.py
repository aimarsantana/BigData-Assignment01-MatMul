#!/usr/bin/env python3
"""Generate the shared input matrices used by all three implementations.

Every implementation (Python, Java, C) reads the SAME binary files, so the
logical inputs are identical across languages (no cross-language RNG issues).

File format: raw little-endian IEEE-754 float64 (binary64), row-major,
n*n values, no header. The size n is encoded in the file name.

Generator : NumPy PCG64 (numpy.random.default_rng)
Seeds     : A -> SEED_A, B -> SEED_B (fixed, see below)
Values    : uniform in [-1, 1)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

SEED_A = 20261012
SEED_B = 20261013
LOW, HIGH = -1.0, 1.0
DEFAULT_SIZES = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]


def make(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(LOW, HIGH, size=(n, n)).astype("<f8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "data", "inputs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    manifest = {"generator": "numpy.random.default_rng (PCG64)", "numpy_version": np.__version__,
                "seed_A": SEED_A, "seed_B": SEED_B, "distribution": f"uniform[{LOW},{HIGH})",
                "dtype": "float64 little-endian, row-major, no header", "files": {}}
    mpath = os.path.join(args.outdir, "manifest.json")
    if os.path.exists(mpath):  # keep entries of previously generated sizes
        with open(mpath) as f:
            manifest["files"].update(json.load(f).get("files", {}))
    for n in args.sizes:
        for name, seed in (("A", SEED_A), ("B", SEED_B)):
            m = make(n, seed)
            path = os.path.join(args.outdir, f"{name}_n{n}.bin")
            m.tofile(path)
            manifest["files"][os.path.basename(path)] = hashlib.sha256(m.tobytes()).hexdigest()
        print(f"generated n={n}", file=sys.stderr)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
