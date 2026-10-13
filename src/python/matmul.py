#!/usr/bin/env python3
"""Basic dense matrix multiplication C = A x B (triple loop, i-j-k order).

Pure Python (no NumPy in the kernel). Storage: list of lists of Python floats
(each row is a list; each element is a boxed float object, IEEE-754 binary64).

Usage:
    python3 matmul.py A.bin B.bin n warmup reps batch [C_out.bin]

Output (stdout, CSV-like):
    INFO,python,<key>,<value>
    SAMPLE,python,<n>,<phase>,<index>,<batch>,<ns_per_multiplication>,<checksum>
The timed region contains ONLY the triple loop (inputs loaded and the
output matrix allocated beforehand). Timer: time.perf_counter_ns (monotonic).
"""
import platform
import sys
import time
from array import array


def read_matrix(path, n):
    buf = array("d")
    with open(path, "rb") as f:
        buf.fromfile(f, n * n)
    if sys.byteorder != "little":
        buf.byteswap()
    return [buf[i * n:(i + 1) * n].tolist() for i in range(n)]


def write_matrix(path, m):
    buf = array("d")
    for row in m:
        buf.extend(row)
    if sys.byteorder != "little":
        buf.byteswap()
    with open(path, "wb") as f:
        buf.tofile(f)


def matmul(A, B, C, n):
    """Straightforward O(n^3) triple loop, i-j-k order, scalar accumulator."""
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s


def peak_rss_kib():
    """Peak resident set size (VmHWM) of this process, from /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def checksum(C):
    return sum(sum(row) for row in C)


def main(argv):
    if len(argv) < 7:
        print(__doc__, file=sys.stderr)
        return 2
    a_path, b_path = argv[1], argv[2]
    n, warmup, reps, batch = (int(x) for x in argv[3:7])
    out_path = argv[7] if len(argv) > 7 else None

    print(f"INFO,python,implementation,{platform.python_implementation()}")
    print(f"INFO,python,version,{platform.python_version()}")
    A = read_matrix(a_path, n)
    B = read_matrix(b_path, n)
    C = [[0.0] * n for _ in range(n)]          # allocated outside the timer

    total = warmup + reps
    for r in range(total):
        phase = "warmup" if r < warmup else "measured"
        idx = r if r < warmup else r - warmup
        t0 = time.perf_counter_ns()
        for _ in range(batch):
            matmul(A, B, C, n)
        t1 = time.perf_counter_ns()
        cs = checksum(C)                       # consume the result (outside timer)
        print(f"SAMPLE,python,{n},{phase},{idx},{batch},{(t1 - t0) / batch:.1f},{cs!r}", flush=True)

    print(f"MEM,python,{n},vmhwm_kib,{peak_rss_kib()}", flush=True)
    if out_path:
        write_matrix(out_path, C)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
