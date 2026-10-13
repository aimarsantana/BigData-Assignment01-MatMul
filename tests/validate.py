#!/usr/bin/env python3
"""Correctness validation for the three implementations (run BEFORE benchmarking).

Test cases
  1. known_2x2        : [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]] (exact)
  2. known_3x3        : hand-checked integer-valued 3x3 product (exact)
  3. identity_left    : I x R = R (exact)
  4. identity_right   : R x I = R (exact)
  5. zero             : 0 x R = 0 (exact)
  6. n=1              : scalar product (exact up to one rounding)
  7. random n in {7, 33, 100, 256} : compared against NumPy (reference library,
     used ONLY for checking) with a componentwise forward-error bound.

Tolerance (random cases)
  For a dot product of length n computed in binary64, the classical bound is
    |C_hat - C| <= gamma_n * (|A| |B|),  gamma_n = n*u / (1 - n*u),  u = 2^-53
  (Higham, Accuracy and Stability of Numerical Algorithms, 2002, Sec. 3.5).
  The NumPy/BLAS reference has its own error of the same order, so we accept
    |C_impl - C_ref| <= 2 * gamma_n * (|A| |B|) + 1e-300
  For exact cases (small integers, identity, zero) we require exact equality.

Cross-language check
  All three kernels perform the same operations in the same order (i-j-k,
  scalar accumulator, no FMA contraction, no reassociation), so their outputs
  are additionally compared for BITWISE equality.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
U = 2.0 ** -53


def runners():
    return {
        "python": [sys.executable, os.path.join(ROOT, "src/python/matmul.py")],
        "java": ["java", "-Xmx4g", "-cp", os.path.join(ROOT, "build"), "MatMul"],
        "c": [os.path.join(ROOT, "build/matmul")],
    }


def run_impl(cmd, A, B, tmp, tag):
    n = A.shape[0]
    pa, pb, pc = (os.path.join(tmp, f"{tag}_{x}.bin") for x in "ABC")
    A.astype("<f8").tofile(pa)
    B.astype("<f8").tofile(pb)
    subprocess.run(cmd + [pa, pb, str(n), "0", "1", "1", pc], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.fromfile(pc, dtype="<f8").reshape(n, n)


def cases():
    rng = np.random.default_rng(12345)
    R4 = rng.uniform(-1, 1, (4, 4))
    yield "known_2x2", np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]]), \
        np.array([[19., 22.], [43., 50.]]), "exact"
    A3 = np.array([[1., 0., 2.], [-1., 3., 1.], [2., 1., 0.]])
    B3 = np.array([[3., 1., 0.], [2., 1., 4.], [1., 0., 5.]])
    C3 = np.array([[5., 1., 10.], [4., 2., 17.], [8., 3., 4.]])
    yield "known_3x3", A3, B3, C3, "exact"
    yield "identity_left", np.eye(4), R4, R4, "exact"
    yield "identity_right", R4, np.eye(4), R4, "exact"
    yield "zero", np.zeros((4, 4)), R4, np.zeros((4, 4)), "exact"
    yield "n1", np.array([[0.5]]), np.array([[-3.0]]), np.array([[-1.5]]), "exact"
    for n in (7, 33, 100, 256):
        A = rng.uniform(-1, 1, (n, n))
        B = rng.uniform(-1, 1, (n, n))
        yield f"random_n{n}", A, B, A @ B, "bound"


def main():
    impls = runners()
    failures = 0
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, A, B, ref, mode in cases():
            n = A.shape[0]
            outs = {}
            for lang, cmd in impls.items():
                C = run_impl(cmd, A, B, tmp, f"{name}_{lang}")
                outs[lang] = C
                if mode == "exact":
                    ok = np.array_equal(C, ref)
                    max_err = float(np.max(np.abs(C - ref)))
                    ratio = float("nan")
                else:
                    gamma = n * U / (1 - n * U)
                    bound = 2 * gamma * (np.abs(A) @ np.abs(B)) + 1e-300
                    err = np.abs(C - ref)
                    ok = bool(np.all(err <= bound))
                    max_err = float(err.max())
                    ratio = float((err / bound).max())
                failures += (not ok)
                rows.append((name, n, lang, mode, max_err, ratio, ok))
            same = all(np.array_equal(outs["c"], outs[l]) for l in ("java", "python"))
            failures += (not same)
            rows.append((name, n, "cross-language", "bitwise", 0.0, float("nan"), same))

    print("case,n,implementation,criterion,max_abs_error,max_error_over_bound,passed")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3e},{r[5]:.3e},{r[6]}")
    print(f"\nTOTAL CHECKS: {len(rows)}  FAILURES: {failures}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
