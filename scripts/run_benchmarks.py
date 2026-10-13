#!/usr/bin/env python3
"""Benchmark harness for Assignment 1.

For every (language, n) configuration it launches ONE fresh process that:
  - loads the shared input files (outside the timer),
  - performs the declared warm-up runs (reported, but excluded from statistics),
  - performs the measured repetitions (each sample = `batch` multiplications,
    reported as time per multiplication).

Memory metric: peak resident set size of the benchmark process (VmHWM, KiB),
read by each implementation from /proc/self/status just before it exits.
VmHWM is reset at execve(), so it describes only the benchmark process (it is
not inflated by this Python harness). It covers the whole process (runtime +
inputs + output + temporaries), not just the matrix arrays.
(os.wait4 ru_maxrss is NOT used: on Linux it carries over the parent's RSS
across fork/exec and would report the harness's ~12 MiB as a floor.)

Budget: if no sample is produced within the per-multiplication budget the
process is killed and the configuration is recorded as TIMEOUT. Larger sizes
for that language are then recorded as NOT_ATTEMPTED (declared stop rule).

Outputs (in results/):
  raw_samples.csv  every warm-up and measured sample
  runs.csv         one row per process: status, peak RSS, wall time
  info.csv         version strings reported by each implementation
  logs/*.log       raw stdout/stderr of every process
"""
import csv
import json
import os
import selectors
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAST_STEAL_S = ""


def load_config(path):
    with open(path) as f:
        return json.load(f)


def reps_for(cfg, n):
    r = cfg["measured_repetitions"]
    return r["n<=512"] if n <= 512 else r["n>512"]


def batch_for(cfg, n):
    b = cfg["batch_per_sample"]
    return int(b.get(str(n), b["default"]))


def command(cfg, lang):
    cmd = list(cfg["commands"][lang])
    if cmd[0] == "python3":
        cmd[0] = sys.executable
    return [c if not c.startswith(("build/", "src/")) else os.path.join(ROOT, c) for c in cmd]


def cpu_steal_ticks():
    """Total 'steal' ticks of all CPUs (time the hypervisor ran other guests), from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        return int(parts[8]) if parts[0] == "cpu" and len(parts) > 8 else None
    except OSError:
        return None


def run_one(cmd, budget_s, log_path):
    """Run a process, stream its stdout, enforce the per-sample budget.
    Returns (status, lines, peak_rss_kib, wall_s, exit_code)."""
    global LAST_STEAL_S
    steal0 = cpu_steal_ticks()
    t_start = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=ROOT)
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    lines, status = [], "OK"
    last = time.monotonic()
    buf = ""
    while True:
        remaining = budget_s - (time.monotonic() - last)
        if remaining <= 0:
            status = "TIMEOUT"
            p.send_signal(signal.SIGKILL)
            break
        events = sel.select(timeout=min(remaining, 5.0))
        if not events:
            continue
        chunk = p.stdout.readline()
        if chunk == "":
            break  # EOF
        lines.append(chunk.rstrip("\n"))
        if chunk.startswith("SAMPLE"):
            last = time.monotonic()
    _, wstatus, _ru = os.wait4(p.pid, 0)
    wall = time.monotonic() - t_start
    exit_code = os.waitstatus_to_exitcode(wstatus)
    p.returncode = exit_code
    if status == "OK" and exit_code != 0:
        status = f"ERROR({exit_code})"
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n" + "\n".join(lines) + "\n")
    steal1 = cpu_steal_ticks()
    hz = os.sysconf("SC_CLK_TCK")
    LAST_STEAL_S = "" if steal0 is None or steal1 is None else f"{(steal1 - steal0) / hz:.2f}"
    rss = ""
    for ln in lines:
        if ln.startswith("MEM"):
            rss = int(ln.split(",")[4])
    return status, lines, rss, wall, exit_code


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "benchmark_config.json"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"),
                    help="output directory (e.g. results/supplementary)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("languages", nargs="*", help="optional subset of languages")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = args.results
    os.makedirs(os.path.join(res, "logs"), exist_ok=True)
    budget = cfg["budget"]["max_seconds_per_single_multiplication"]
    only = args.languages

    samples_f = open(os.path.join(res, "raw_samples.csv"), "w", newline="")
    runs_f = open(os.path.join(res, "runs.csv"), "w", newline="")
    info_f = open(os.path.join(res, "info.csv"), "w", newline="")
    sw, rw, iw = csv.writer(samples_f), csv.writer(runs_f), csv.writer(info_f)
    sw.writerow(["language", "n", "numeric_type", "threads", "phase", "index", "batch", "ns_per_multiplication", "checksum"])
    rw.writerow(["language", "n", "numeric_type", "threads", "kind", "status", "warmup_runs", "measured_reps", "batch",
                 "budget_s_per_multiplication", "peak_rss_kib", "process_wall_s", "exit_code",
                 "cpu_steal_s"])
    iw.writerow(["language", "key", "value"])

    stopped = set()
    langs = [l for l in cfg["languages"] if not only or l in only]

    # 1) Runtime baseline: load the n=64 inputs, perform no multiplication.
    for lang in ([] if args.no_baseline else langs):
        a = os.path.join(ROOT, "data", "inputs", "A_n64.bin")
        b = os.path.join(ROOT, "data", "inputs", "B_n64.bin")
        cmd = command(cfg, lang) + [a, b, "64", "0", "0", "1"]
        st, lines, rss, wall, ec = run_one(cmd, budget, os.path.join(res, "logs", f"{lang}_baseline.log"))
        rw.writerow([lang, 64, "float64", 1, "baseline_no_multiplication", st, 0, 0, 1, budget, rss, f"{wall:.3f}", ec, LAST_STEAL_S])
        for ln in lines:
            if ln.startswith("INFO"):
                iw.writerow(ln.split(",", 3)[1:])
        print(f"[baseline] {lang}: peak RSS {rss} KiB", flush=True)

    # 2) Measurements, increasing n.
    for n in cfg["sizes"]:
        for lang in langs:
            warm, reps, batch = cfg["warmup_runs"][lang], reps_for(cfg, n), batch_for(cfg, n)
            if lang in stopped:
                rw.writerow([lang, n, "float64", 1, "benchmark", "NOT_ATTEMPTED", warm, reps, batch, budget, "", "", ""])
                runs_f.flush()
                print(f"[n={n}] {lang}: not attempted (stop rule)", flush=True)
                continue
            a = os.path.join(ROOT, "data", "inputs", f"A_n{n}.bin")
            b = os.path.join(ROOT, "data", "inputs", f"B_n{n}.bin")
            cmd = command(cfg, lang) + [a, b, str(n), str(warm), str(reps), str(batch)]
            st, lines, rss, wall, ec = run_one(cmd, budget * batch,
                                               os.path.join(res, "logs", f"{lang}_n{n}.log"))
            for ln in lines:
                if ln.startswith("SAMPLE"):
                    f = ln.split(",")[1:]
                    sw.writerow(f[:2] + ["float64", 1] + f[2:])
            rw.writerow([lang, n, "float64", 1, "benchmark", st, warm, reps, batch, budget, rss, f"{wall:.3f}", ec,
                         LAST_STEAL_S])
            samples_f.flush(); runs_f.flush()
            if st != "OK":
                stopped.add(lang)
            if rss != "" and rss * 1024 > cfg["budget"]["memory_budget_peak_rss_bytes"]:
                stopped.add(lang)
            print(f"[n={n}] {lang}: {st}, peak RSS {rss} KiB, wall {wall:.1f}s", flush=True)

    for f in (samples_f, runs_f, info_f):
        f.close()


if __name__ == "__main__":
    main()
