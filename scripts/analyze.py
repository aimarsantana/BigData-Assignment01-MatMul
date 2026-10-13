#!/usr/bin/env python3
"""Turn the raw measurements into summary tables, LaTeX tables and figures.

Inputs : results/raw_samples.csv, results/runs.csv
Outputs: results/summary.csv, results/scaling_fit.csv, results/ratios.csv,
         figures/*.pdf + *.png, report/generated/*.tex
Only phase == "measured" samples enter the statistics; warm-up samples are
plotted separately (fig_warmup) and never mixed into the summaries.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, FIG, GEN = (os.path.join(ROOT, d) for d in ("results", "figures", "report/generated"))
LANGS = ["c", "java", "python"]
LABEL = {"c": "C (gcc -O2)", "java": "Java (OpenJDK 21, HotSpot)", "python": "Python (CPython 3.13)"}
COLOR = {"c": "#1f77b4", "java": "#d95f02", "python": "#2a9d4b"}
MARK = {"c": "o", "java": "s", "python": "^"}
FIT_MIN_N = 256  # sizes used for the log-log slope (avoid tiny, overhead-dominated sizes)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(GEN, exist_ok=True)
    raw = pd.read_csv(os.path.join(RES, "raw_samples.csv"))
    runs = pd.read_csv(os.path.join(RES, "runs.csv"))
    raw["seconds"] = raw["ns_per_multiplication"] / 1e9
    meas = raw[raw["phase"] == "measured"]

    # ---------- summary per configuration ----------
    g = meas.groupby(["language", "n"])["seconds"]
    summ = pd.DataFrame({
        "reps": g.count(),
        "median_s": g.median(),
        "q1_s": g.quantile(0.25),
        "q3_s": g.quantile(0.75),
        "min_s": g.min(),
        "max_s": g.max(),
        "mean_s": g.mean(),
        "std_s": g.std(ddof=1),
    }).reset_index()
    summ["iqr_s"] = summ["q3_s"] - summ["q1_s"]
    summ["rel_iqr_pct"] = 100 * summ["iqr_s"] / summ["median_s"]
    summ["ns_per_inner_iter"] = summ["median_s"] * 1e9 / summ["n"] ** 3
    summ["mflops"] = 2 * summ["n"] ** 3 / summ["median_s"] / 1e6
    bench = runs[runs["kind"] == "benchmark"][["language", "n", "status", "batch", "peak_rss_kib"]]
    summ = summ.merge(bench, on=["language", "n"], how="left")
    summ["peak_rss_mib"] = summ["peak_rss_kib"] / 1024
    summ["arrays_3x8n2_mib"] = 3 * 8 * summ["n"] ** 2 / 2 ** 20
    summ = summ.sort_values(["language", "n"])
    summ.to_csv(os.path.join(RES, "summary.csv"), index=False, float_format="%.6g")

    # ---------- log-log slope (empirical growth exponent) ----------
    fits = []
    for lang in LANGS:
        s = summ[(summ.language == lang) & (summ.n >= FIT_MIN_N)]
        if len(s) >= 2:
            b, a = np.polyfit(np.log(s.n), np.log(s.median_s), 1)
            pred = np.exp(a) * s.n ** b
            r2 = 1 - np.sum((np.log(s.median_s) - np.log(pred)) ** 2) / np.sum(
                (np.log(s.median_s) - np.log(s.median_s).mean()) ** 2)
            fits.append({"language": lang, "n_min": int(s.n.min()), "n_max": int(s.n.max()),
                         "points": len(s), "slope": b, "r2": r2})
    fits = pd.DataFrame(fits)
    fits.to_csv(os.path.join(RES, "scaling_fit.csv"), index=False, float_format="%.4f")

    # ---------- ratios relative to C ----------
    piv = summ.pivot(index="n", columns="language", values="median_s")
    ratios = pd.DataFrame({"n": piv.index})
    for lang in ("java", "python"):
        if lang in piv and "c" in piv:
            ratios[f"{lang}_over_c"] = (piv[lang] / piv["c"]).values
    ratios.to_csv(os.path.join(RES, "ratios.csv"), index=False, float_format="%.3f")

    # ---------- figure 1: runtime vs n (log-log) ----------
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for lang in LANGS:
        s = summ[summ.language == lang]
        if s.empty:
            continue
        ax.errorbar(s.n, s.median_s, yerr=[s.median_s - s.q1_s, s.q3_s - s.median_s],
                    marker=MARK[lang], color=COLOR[lang], label=LABEL[lang], capsize=2, lw=1.2, ms=4)
        ref = s[s.n == s.n.min()]
        nn = np.array(sorted(summ.n.unique()))
        ax.plot(nn, ref.median_s.values[0] * (nn / ref.n.values[0]) ** 3, ls=":", color=COLOR[lang], lw=0.9)
    budget_s = runs.budget_s_per_multiplication.max()
    to = runs[(runs.kind == "benchmark") & (runs.status == "TIMEOUT")]
    for k, r in enumerate(to.itertuples()):
        ax.plot(r.n, budget_s, marker="x", ms=8, mew=2, color=COLOR[r.language], ls="none",
                label=f"timeout: > {int(budget_s)} s budget (no time recorded)" if k == 0 else None)
    ax.axhline(budget_s, color="grey", lw=0.6, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (n x n)")
    ax.set_ylabel("Time per multiplication [s] (median, IQR bars)")
    ax.set_title("Runtime vs. matrix size (dotted: n$^3$ from smallest n)")
    ax.legend()
    save(fig, "fig_runtime_loglog")

    # ---------- figure 2: normalised time per inner iteration ----------
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for lang in LANGS:
        s = summ[summ.language == lang]
        if s.empty:
            continue
        ax.plot(s.n, s.ns_per_inner_iter, marker=MARK[lang], color=COLOR[lang], label=LABEL[lang], lw=1.2, ms=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Median time / n$^3$ [ns per inner iteration]")
    ax.set_title("Cost of one multiply-add (constant if growth were exactly cubic)")
    ax.legend()
    save(fig, "fig_time_per_iteration")

    # ---------- figure 3: memory ----------
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    base = runs[runs.kind == "baseline_no_multiplication"].set_index("language")["peak_rss_kib"] / 1024
    for lang in LANGS:
        s = summ[summ.language == lang]
        if s.empty:
            continue
        ax.plot(s.n, s.peak_rss_mib, marker=MARK[lang], color=COLOR[lang], label=f"{LABEL[lang]} peak RSS", lw=1.2, ms=4)
    nn = np.array(sorted(summ.n.unique()))
    ax.plot(nn, 3 * 8 * nn ** 2 / 2 ** 20, "k--", lw=1, label="3 x 8n$^2$ bytes (raw float64 arrays)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Peak resident set size [MiB]")
    ax.set_title("Peak process memory (VmHWM) vs. matrix size")
    ax.legend(fontsize=7)
    save(fig, "fig_memory")

    # ---------- figure 4: Java warm-up / JIT ----------
    jw = raw[raw.language == "java"].copy()
    if not jw.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        for n, col in zip([128, 512, 1024], ["#555555", "#d95f02", "#7b3294"]):
            s = jw[jw.n == n].copy()
            if s.empty:
                continue
            s["order"] = np.where(s.phase == "warmup", s["index"], s["index"] + (s.phase == "warmup").sum())
            s = s.sort_values("order")
            med = s[s.phase == "measured"].seconds.median()
            ax.plot(range(1, len(s) + 1), s.seconds / med, marker="o", ms=3, color=col, label=f"n={n}")
            nw = (s.phase == "warmup").sum()
        ax.axvline(nw + 0.5, color="k", lw=0.8, ls="--")
        ax.text(nw + 0.6, ax.get_ylim()[1] * 0.95, "measured →", va="top", fontsize=8)
        ax.set_xlabel("Run number within the JVM process (warm-up runs first)")
        ax.set_ylabel("Time / median of measured runs")
        ax.set_title("Java: effect of JIT warm-up (warm-up runs excluded from statistics)")
        ax.legend()
        save(fig, "fig_java_warmup")

    # ---------- figure 5: ratio to C ----------
    if len(ratios.columns) > 1:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        for lang in ("java", "python"):
            col = f"{lang}_over_c"
            if col in ratios:
                ax.plot(ratios.n, ratios[col], marker=MARK[lang], color=COLOR[lang], label=f"{LABEL[lang]} / C", lw=1.2, ms=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xlabel("Matrix size n")
        ax.set_ylabel("Median time ratio relative to C")
        ax.set_title("Relative runtime (same algorithm, inputs and machine)")
        ax.legend()
        save(fig, "fig_ratio_to_c")

    # ---------- LaTeX tables ----------
    def fmt_t(x):
        if x < 1e-3:
            return f"{x*1e6:.1f}\\,\\textmu s"
        if x < 1:
            return f"{x*1e3:.2f}\\,ms"
        return f"{x:.2f}\\,s"

    lines = []
    for lang in LANGS:
        s = summ[summ.language == lang]
        for _, r in s.iterrows():
            lines.append(f"{LABEL[lang].split(' (')[0]} & {int(r.n)} & {int(r.reps)}"
                         f"{'' if r.batch == 1 else f' ($\\times${int(r.batch)})'} & {fmt_t(r.median_s)} & "
                         f"{fmt_t(r.iqr_s)} ({r.rel_iqr_pct:.1f}\\%) & {fmt_t(r.min_s)} & {fmt_t(r.max_s)} & "
                         f"{r.ns_per_inner_iter:.2f} & {r.mflops:.0f} \\\\")
        # timeouts / not attempted
        for _, r in runs[(runs.language == lang) & (runs.kind == "benchmark") & (runs.status != "OK")].iterrows():
            lines.append(f"{LABEL[lang].split(' (')[0]} & {int(r.n)} & -- & \\multicolumn{{6}}{{l}}"
                         f"{{\\textit{{{r.status.replace('_', ' ').lower()}}} (budget {int(r.budget_s_per_multiplication)}\\,s per multiplication)}} \\\\")
        lines.append("\\midrule")
    open(os.path.join(GEN, "table_time.tex"), "w").write("\n".join(lines[:-1]) + "\n")

    mlines = []
    for n in sorted(summ.n.unique()):
        row = [str(int(n)), f"{3*8*n**2/2**20:.1f}"]
        for lang in LANGS:
            v = summ[(summ.language == lang) & (summ.n == n)].peak_rss_mib
            row.append(f"{v.values[0]:.1f}" if len(v) and not np.isnan(v.values[0]) else "--")
        mlines.append(" & ".join(row) + " \\\\")
    brow = ["baseline", "--"] + [f"{base.get(l, float('nan')):.1f}" for l in LANGS]
    mlines.insert(0, " & ".join(brow) + " \\\\ \\midrule")
    open(os.path.join(GEN, "table_memory.tex"), "w").write("\n".join(mlines) + "\n")

    flines = [f"{LABEL[r.language].split(' (')[0]} & {r.n_min}--{r.n_max} & {r.points} & {r.slope:.2f} & {r.r2:.4f} \\\\"
              for r in fits.itertuples()]
    open(os.path.join(GEN, "table_fit.tex"), "w").write("\n".join(flines) + "\n")

    rlines = []
    for r in ratios.itertuples():
        j = getattr(r, "java_over_c", float("nan"))
        p = getattr(r, "python_over_c", float("nan"))
        rlines.append(f"{r.n} & {'--' if np.isnan(j) else f'{j:.2f}'} & {'--' if np.isnan(p) else f'{p:.1f}'} \\\\")
    open(os.path.join(GEN, "table_ratio.tex"), "w").write("\n".join(rlines) + "\n")

    print(summ[["language", "n", "reps", "median_s", "iqr_s", "rel_iqr_pct", "ns_per_inner_iter", "peak_rss_mib"]].to_string(index=False))
    print(fits.to_string(index=False))
    print(ratios.to_string(index=False))


if __name__ == "__main__":
    main()
