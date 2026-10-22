#!/usr/bin/env python3
"""Render the README figures from the raw result files (needs matplotlib; runs in the jupyter image).

  python3 analysis/make_figures.py            -> docs/figures/*.png

Sources: results/*.json and *.telemetry.csv (trianglebench), peak/results/*/*.txt (peak tools).
Every number in a figure is read from these files; nothing is typed in by hand except the
datasheet / derived theoretical peaks, which are labelled as such.
"""
import csv, os, re, statistics, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import summarize  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
C_A4000, C_GB10, C_XEON, C_GBCPU = "#1f77b4", "#76b900", "#7f7f7f", "#2ca02c"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

NAMES = {"M1_cuda_int32_bitset_popc": "M1 CUDA INT32\nbitset+popc", "M2_cuda_fp32_gemm": "M2 CUDA FP32\nGEMM",
         "M3_tensor_fp16_fp32acc_gemm": "M3 Tensor\nFP16->FP32", "M4_tensor_int8_int32acc_gemm": "M4 Tensor\nINT8->INT32",
         "P_all_four_parallel": "all four\nin parallel", "C1_cpu_int64_bitset_popc": "C1 CPU INT64\nbitset+popc",
         "C2_cpu_fp32_gemm_blas": "C2 CPU FP32\nGEMM (BLAS)"}


def load(files):
    """mean, 95 % CI half-width and per-window means per (file, n, method)."""
    g = defaultdict(list)
    for f in files:
        for r in summarize.rows(os.path.join(ROOT, "results", f)):
            g[(f, r["n"], r["method"])].append(r)
    out = {}
    for k, rs in g.items():
        def m(col):
            v = [r[col] for r in rs if r[col] not in (None, "")]
            return statistics.mean(v) if v else None
        out[k] = {"tops": m("eq_tops"), "ci": summarize.ci95([r["eq_tops"] for r in rs]), "w": m("watts"),
                  "j": m("joule_per_solve"), "t": max(r["temp_max_c"] or 0 for r in rs) or None,
                  "mhz": statistics.median([r["sm_mhz_median"] for r in rs if r["sm_mhz_median"]] or [0]) or None}
    return out


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote docs/figures/" + name)


def fig_throughput(d):
    """eq.TOPS of every method on every system, n = 8192, log axis."""
    rows = []
    for f, lab, col, meths in [("xeon-w2125_cpu.json", "Xeon W-2125 CPU", C_XEON, ["C2_cpu_fp32_gemm_blas", "C1_cpu_int64_bitset_popc"]),
                               ("dgx-spark-gb10_cpu.json", "GB10 CPU (20 cores)", C_GBCPU, ["C2_cpu_fp32_gemm_blas", "C1_cpu_int64_bitset_popc"]),
                               ("rtx-a4000_std.json", "RTX A4000", C_A4000, ["M2_cuda_fp32_gemm", "P_all_four_parallel", "M3_tensor_fp16_fp32acc_gemm", "M1_cuda_int32_bitset_popc", "M4_tensor_int8_int32acc_gemm"]),
                               ("dgx-spark-gb10_std.json", "DGX Spark GB10", C_GB10, ["M2_cuda_fp32_gemm", "P_all_four_parallel", "M3_tensor_fp16_fp32acc_gemm", "M1_cuda_int32_bitset_popc", "M4_tensor_int8_int32acc_gemm"])]:
        for m in meths:
            v = d[(f, 8192, m)]
            rows.append((f"{lab}: {NAMES[m].replace(chr(10), ' ')}", v["tops"], v["ci"], col))
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    y = range(len(rows))
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color=[r[3] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[1] * 1.06, i, f"{r[1]:.1f}" if r[1] >= 1 else f"{r[1]:.2f}", va="center", fontsize=8)
    ax.set_yticks(list(y), [r[0] for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 400)
    ax.set_xlabel("eq.TOPS = 2n³ / time  (log scale, mean ± 95 % CI, 10 x 5 s)")
    ax.set_title("Same problem (triangles, n = 8192), 14 ways: CPU vs. CUDA cores vs. Tensor Cores")
    save(fig, "throughput.png")


def fig_efficiency(d):
    """throughput vs. power with iso-efficiency lines; marker colour = max. temperature."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for w in (0.1, 0.25, 0.5, 1, 2):
        ax.plot([0, 160], [0, 160 * w], color="0.8", lw=0.8, zorder=0)
        xl = min(155, 145 / w)
        ax.text(xl, xl * w + 2, f"{w} TOPS/W", color="0.5", fontsize=8, ha="right")
    pts = []
    for f, mk, lab in [("rtx-a4000_std.json", "s", "RTX A4000"), ("dgx-spark-gb10_std.json", "o", "DGX Spark GB10"),
                       ("xeon-w2125_cpu.json", "^", "Xeon W-2125 (RAPL)")]:
        for (ff, n, m), v in d.items():
            if ff != f or n != 8192 or v["w"] is None:
                continue
            pts.append((v["w"], v["tops"], v["t"], mk))
            ax.annotate(m.split("_")[0], (v["w"], v["tops"]), textcoords="offset points", xytext=(6, -3), fontsize=8)
        ax.scatter([], [], marker=mk, color="0.4", label=lab)
    sc = None
    for mk in "so^":
        p = [x for x in pts if x[3] == mk]
        sc = ax.scatter([x[0] for x in p], [x[1] for x in p], c=[x[2] for x in p], cmap="coolwarm", vmin=75, vmax=90,
                        marker=mk, s=70, edgecolor="k", zorder=3)
    fig.colorbar(sc, ax=ax, label="max. temperature °C")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 150)
    ax.set_xlabel("median power during the window, W (GPU: NVML, CPU: package RAPL)")
    ax.set_ylabel("eq.TOPS")
    ax.set_title("Throughput vs. power, n = 8192 (GB10 CPU has no energy counter)")
    ax.legend(loc="upper left")
    save(fig, "efficiency.png")


def fig_energy(d):
    """joule per solved graph, n = 8192, log axis."""
    rows = []
    for f, lab, col in [("xeon-w2125_cpu.json", "Xeon", C_XEON), ("rtx-a4000_std.json", "A4000", C_A4000), ("dgx-spark-gb10_std.json", "GB10", C_GB10)]:
        for (ff, n, m), v in sorted(d.items()):
            if ff == f and n == 8192 and v["j"]:
                rows.append((f"{lab} {m.split('_')[0]}", v["j"], col))
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, r[1] * 1.1, f"{r[1]:.2f}" if r[1] < 10 else f"{r[1]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("joule per solved graph (log)")
    ax.set_title("Energy to count all triangles of one 8192-node graph")
    save(fig, "energy.png")


def fig_telemetry():
    """raw NVML samples of repetition 1 at n = 8192: power, temperature, SM clock per method window."""
    fig, axs = plt.subplots(3, 2, figsize=(10, 6.5), sharex="col", sharey="row")
    order = ["M1_cuda_int32_bitset_popc", "M2_cuda_fp32_gemm", "M3_tensor_fp16_fp32acc_gemm", "M4_tensor_int8_int32acc_gemm", "all"]
    for c, (f, lab, col) in enumerate([("rtx-a4000_std.telemetry.csv", "RTX A4000 (power cap 140 W)", C_A4000),
                                       ("dgx-spark-gb10_std.telemetry.csv", "DGX Spark GB10", C_GB10)]):
        w = defaultdict(list)
        with open(os.path.join(ROOT, "results", f)) as fh:
            for r in csv.DictReader(fh):
                if r["n"] == "8192" and r["repeat"] == "1":
                    w[r["method"]].append(r)
        off = 0.0
        for m in order:
            rs = w.get(m, [])
            t = [off + float(r["t_s"]) for r in rs]
            for i, key in enumerate(("power_w", "temp_c", "sm_mhz")):
                axs[i][c].plot(t, [float(r[key]) if r[key] else None for r in rs], color=col, lw=1)
            if rs:
                axs[0][c].text(off + 2.5, 2, (m.split("_")[0] if m != "all" else "P"), ha="center", fontsize=8)
                for i in range(3):
                    axs[i][c].axvline(off, color="0.85", lw=0.8)
                off = t[-1] + 0.5
        axs[0][c].set_title(lab)
        axs[2][c].set_xlabel("time within repetition 1, s (5 s window per method)")
    axs[0][0].set_ylabel("power W")
    axs[1][0].set_ylabel("temperature °C")
    axs[2][0].set_ylabel("SM clock MHz")
    axs[0][0].set_ylim(0, 150)
    fig.suptitle("Raw telemetry (100 ms NVML samples), n = 8192")
    save(fig, "telemetry.png")


def fig_quant(d):
    """input formats in the --quant study: throughput and efficiency."""
    fmts = ["fp32", "tf32_fp32", "fp16_fp32", "bf16_fp32", "fp8e4m3_fp32", "int8_int32"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    x = range(len(fmts))
    for k, (f, lab, col, dx) in enumerate([("rtx-a4000_quant.json", "RTX A4000", C_A4000, -0.2), ("dgx-spark-gb10_quant.json", "GB10", C_GB10, 0.2)]):
        v = [d.get((f, 8192, "Q_" + q)) for q in fmts]
        for a, key in zip(axs, ("tops", "eff")):
            ys = [(e["tops"] if key == "tops" else e["tops"] / e["w"]) if e else 0 for e in v]
            a.bar([i + dx for i in x], ys, width=0.4, color=col, label=lab)
            for i, yv in enumerate(ys):
                a.text(i + dx, yv, "n/a" if not yv else (f"{yv:.0f}" if key == "tops" else f"{yv:.2f}"), ha="center", va="bottom", fontsize=7)
    for a, t in zip(axs, ("eq.TOPS", "eq.TOPS per W")):
        a.set_xticks(list(x), [q.split("_")[0] for q in fmts])
        a.set_ylabel(t)
    axs[0].legend()
    fig.suptitle("Optional --quant study, n = 8192: every format returns the exact count; only speed and energy differ")
    save(fig, "quant.png")


def peak_tel(path):
    """telemetry table of a peak log: name -> (value, Tmax, P median, MHz)."""
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 6 and re.match(r"^[a-z_0-9]+_\d+$", p[0]) and re.match(r"^[\d.]+$", p[1]) and re.match(r"^\d+$", p[2]):
            out[p[0]] = (float(p[1]), int(p[2]), float(p[3]), int(p[5]))
    return out


def fig_peak():
    """measured peak vs. theory at the measured clock and vs. the marketing figure."""
    g = {**peak_tel(os.path.join(ROOT, "peak/results/dgx-spark-gb10/dense-tel.txt")),
         **peak_tel(os.path.join(ROOT, "peak/results/dgx-spark-gb10/sparse-tel.txt"))}
    # derived per-SM rates (FLOP/clock/SM, dense), 48 SMs; x2 for 2:4 sparsity - see peak/README.md
    rows = [("FP16", "fp16_8192", 1024, 1), ("INT8", "int8_8192", 2048, 1), ("FP8", "fp8_8192", 2048, 1), ("NVFP4", "nvfp4_8192", 4096, 1),
            ("FP16\n2:4", "sp_fp16_8192", 1024, 2), ("INT8\n2:4", "sp_int8_8192", 2048, 2), ("FP8\n2:4", "sp_fp8_8192", 2048, 2),
            ("FP4\n2:4", "sp_fp4_8192", 4096, 2), ("FP4 2:4\nn=16384", "sp_fp4_16384", 4096, 2)]
    a = peak_tel(os.path.join(ROOT, "peak/results/rtx-a4000/dense.txt"))
    a.update(peak_tel(os.path.join(ROOT, "peak/results/rtx-a4000/sparse.txt")))
    a.update({"c_" + k: v for k, v in peak_tel(os.path.join(ROOT, "peak/results/rtx-a4000/cutlass.txt")).items()})
    arows = [("FP16", "fp16_8192", 76.7), ("INT8", "int8_8192", 153.4), ("INT4\n(CUTLASS)", "c_int4_8192", 306.8),
             ("FP16\n2:4", "sp_fp16_8192", 153.4), ("INT8\n2:4", "sp_int8_8192", 306.8), ("INT4 2:4\n(CUTLASS)", "c_sp_int4_8192", 613.4)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    ax = axs[0]
    th = [48 * r * f * g[k][3] * 1e6 / 1e12 for _, k, r, f in rows]
    me = [g[k][0] for _, k, _, _ in rows]
    x = range(len(rows))
    ax.bar(x, th, color="0.85", label="theory at measured clock (derived)")
    ax.bar(x, me, width=0.55, color=C_GB10, label="measured (cuBLASLt / cuSPARSELt)")
    ax.axhline(1000, color="crimson", ls="--", lw=1)
    ax.text(-0.4, 1010, "NVIDIA: 'up to 1 PFLOP FP4 with sparsity'", color="crimson", fontsize=8, va="bottom")
    for i, (m, t) in enumerate(zip(me, th)):
        ax.text(i, m + 8, f"{m:.0f}\n{100 * m / t:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(list(x), [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1100)
    ax.set_ylabel("TFLOPS / TOPS (effective for 2:4)")
    ax.set_title("DGX Spark GB10, n = 8192, GPU ~92 W")
    ax.legend(loc="center left", fontsize=8)
    ax = axs[1]
    x = range(len(arows))
    ax.bar(x, [r[2] for r in arows], color="0.85", label="datasheet (1560 MHz boost)")
    ax.bar(x, [a[k][0] for _, k, _ in arows], width=0.55, color=C_A4000, label="measured at 140 W cap")
    for i, (_, k, ds) in enumerate(arows):
        ax.text(i, a[k][0] + 8, f"{a[k][0]:.0f}\n{100 * a[k][0] / ds:.0f}%\n{a[k][3]} MHz", ha="center", fontsize=7)
    ax.set_xticks(list(x), [r[0] for r in arows], fontsize=8)
    ax.set_title("RTX A4000, n = 8192")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Peak tools (peak/): how close do the GPUs get to their datasheet?")
    save(fig, "peak.png")


def fig_fused():
    """fused CUDA Tile kernel vs. cuBLASLt + epilogue over n."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for path, lab, col in [("peak/results/dgx-spark-gb10/fusedtri-tel.txt", "GB10", C_GB10), ("peak/results/rtx-a4000/fusedtri.txt", "RTX A4000", C_A4000)]:
        t = peak_tel(os.path.join(ROOT, path))
        ns = sorted({int(k.split("_")[1]) for k in t})
        ax.plot(ns, [t[f"unfused_{n}"][0] for n in ns], "-o", color=col, label=f"{lab} cuBLASLt GEMM + epilogue")
        ax.plot(ns, [t[f"fused_{n}"][0] for n in ns], "--s", color=col, label=f"{lab} fused CUDA Tile kernel")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4096, 8192, 16384], ["4096", "8192", "16384"])
    ax.set_xlabel("nodes n")
    ax.set_ylabel("eq.TOPS (INT8)")
    ax.set_title("Fused CUDA Tile kernel vs. cuBLASLt + epilogue (INT8, density 0.05)")
    ax.set_ylim(0, 200)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "fused.png")


def main():
    files = [f for f in os.listdir(os.path.join(ROOT, "results")) if f.endswith(".json")]
    d = load(files)
    fig_throughput(d)
    fig_efficiency(d)
    fig_energy(d)
    fig_telemetry()
    fig_quant(d)
    fig_peak()
    fig_fused()


if __name__ == "__main__":
    main()
