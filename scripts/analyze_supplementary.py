#!/usr/bin/env python3
"""Analyse the supplementary power-of-two experiment (C and Java only).

Input : results/supplementary/raw_samples.csv
Output: results/supplementary/summary.csv, figures/fig_supp_alignment.{pdf,png},
        report/generated/table_supp.tex
This experiment is reported separately from the main controlled comparison.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results", "supplementary")
FIG = os.path.join(ROOT, "figures")
GEN = os.path.join(ROOT, "report", "generated")


def main():
    raw = pd.read_csv(os.path.join(RES, "raw_samples.csv"))
    m = raw[raw.phase == "measured"].copy()
    m["seconds"] = m.ns_per_multiplication / 1e9
    g = m.groupby(["language", "n"]).seconds
    s = pd.DataFrame({"reps": g.count(), "median_s": g.median(),
                      "q1_s": g.quantile(.25), "q3_s": g.quantile(.75)}).reset_index()
    s["iqr_s"] = s.q3_s - s.q1_s
    s["ns_per_inner_iter"] = s.median_s * 1e9 / s.n ** 3
    s.to_csv(os.path.join(RES, "summary.csv"), index=False, float_format="%.6g")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, group in zip(axes, ([500, 511, 512, 513], [1000, 1023, 1024, 1025])):
        sub = s[s.n.isin(group)]
        width = 0.38
        for k, (lang, col) in enumerate((("c", "#1f77b4"), ("java", "#d95f02"))):
            d = sub[sub.language == lang].set_index("n").reindex(group)
            x = [i + (k - 0.5) * width for i in range(len(group))]
            ax.bar(x, d.ns_per_inner_iter, width, color=col, label={"c": "C", "java": "Java"}[lang])
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([str(n) for n in group])
        ax.set_xlabel("n")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Median time / n$^3$ [ns]")
    axes[0].legend()
    fig.suptitle("Supplementary: sizes around powers of two (same kernel)", fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_supp_alignment.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_supp_alignment.png"), bbox_inches="tight", dpi=150)

    piv = s.pivot(index="n", columns="language", values="ns_per_inner_iter")
    lines = [f"{n} & {piv.loc[n, 'c']:.2f} & {piv.loc[n, 'java']:.2f} \\\\" for n in piv.index]
    open(os.path.join(GEN, "table_supp.tex"), "w").write("\n".join(lines) + "\n")
    print(s.to_string(index=False))

    # Experiment B: extended-budget probe at n = 2047 / 2048
    lp = os.path.join(ROOT, "results", "supplementary_large", "raw_samples.csv")
    if os.path.exists(lp):
        r = pd.read_csv(lp)
        r = r[r.phase == "measured"].copy()
        r["seconds"] = r.ns_per_multiplication / 1e9
        g = r.groupby(["language", "n"]).seconds
        L = pd.DataFrame({"reps": g.count(), "median_s": g.median(), "min_s": g.min(),
                          "max_s": g.max()}).reset_index()
        L["ns_per_inner_iter"] = L.median_s * 1e9 / L.n ** 3
        L.to_csv(os.path.join(ROOT, "results", "supplementary_large", "summary.csv"),
                 index=False, float_format="%.6g")
        lines = [f"{'C' if x.language == 'c' else 'Java'} & {x.n} & {x.reps} & {x.median_s:.1f} & "
                 f"{x.min_s:.1f}--{x.max_s:.1f} & {x.ns_per_inner_iter:.2f} \\\\" for x in L.itertuples()]
        open(os.path.join(GEN, "table_supp_large.tex"), "w").write("\n".join(lines) + "\n")
        print(L.to_string(index=False))


if __name__ == "__main__":
    main()
