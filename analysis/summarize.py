#!/usr/bin/env python3
"""Combine trianglebench result files for statistical analysis (Python standard library only).

  python3 analysis/summarize.py results/*.json                 -> summary table per system, size and method
  python3 analysis/summarize.py results/*.json --csv all.csv   -> tidy CSV, one row per measuring window
  python3 analysis/summarize.py a.json b.json --compare        -> ratio b/a per method with Welch t-test

The tidy CSV has one row per (system, n, method, repetition) and can be loaded directly with
pandas.read_csv() or R read.csv() for ANOVA, mixed models, plots etc.
GPU files (trianglebench) and CPU files (trianglebench-cpu) can be mixed: column device_type is "gpu" or "cpu",
column gpu holds the device name, sm_mhz_median the median SM clock (GPU) or median core clock (CPU).
For CPUs, watts is the RAPL package power (empty where the platform has no CPU energy counter).
"""
import argparse, csv, json, math, statistics, sys

FIELDS = ["file", "label", "device_type", "gpu", "machine", "driver", "cuda_runtime", "power_limit_w", "tool_version", "git_commit",
          "n", "p", "seed", "method", "repeat", "position_in_order", "solves", "window_s", "seconds_per_solve",
          "eq_tops", "watts", "energy_j", "joule_per_solve", "eq_tops_per_watt", "temp_max_c", "sm_mhz_median",
          "mem_mhz_median", "clock_event_reasons_or", "energy_source", "verified"]


def div(a, b):
    return a / b if a is not None and b else None


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else float("nan")


def cpu_rows(path, d):
    c, t, par = d["cpu"], d["tool"], d["parameters"]
    for run in d["runs"]:
        for m in run["methods"]:
            for r in m["repetitions"]:
                yield {"file": path, "label": d["label"], "device_type": "cpu", "gpu": c["name"],
                       "machine": d["system"]["machine"], "driver": None, "cuda_runtime": None, "power_limit_w": None,
                       "tool_version": t["version"], "git_commit": t["git_commit"], "n": run["n"], "p": par["p"],
                       "seed": par["seed"], "method": m["id"], "repeat": r["repeat"],
                       "position_in_order": r["position_in_order"], "solves": r["solves"], "window_s": r["window_s"],
                       "seconds_per_solve": r["seconds_per_solve"], "eq_tops": r["eq_tops"], "watts": r["watts"],
                       "energy_j": r["energy_j"], "joule_per_solve": r["joule_per_solve"],
                       "eq_tops_per_watt": div(r["eq_tops"], r["watts"]), "temp_max_c": r["temp_max_c"],
                       "sm_mhz_median": r["cpu_mhz_median"], "mem_mhz_median": None,
                       "clock_event_reasons_or": r["thermal_throttle_events"], "energy_source": r["energy_source"],
                       "verified": m["verified"]}


def rows(path):
    d = json.load(open(path))
    schema = str(d.get("schema", ""))
    if schema.startswith("trianglebench-cpu-result/"):
        yield from cpu_rows(path, d)
        return
    if not schema.startswith("trianglebench-result/"):
        sys.exit(f"{path}: not a trianglebench result file")
    g, t, par = d["gpu"], d["tool"], d["parameters"]
    for run in d["runs"]:
        for m in run["methods"]:
            for r in m["repetitions"]:
                w = r["watts"]
                yield {"file": path, "label": d["label"], "device_type": "gpu", "gpu": g["name"], "machine": d["system"]["machine"],
                       "driver": g["driver"], "cuda_runtime": g["cuda_runtime"], "power_limit_w": g["power_limit_w"],
                       "tool_version": t["version"], "git_commit": t["git_commit"], "n": run["n"], "p": par["p"],
                       "seed": par["seed"], "method": m["id"], "repeat": r["repeat"],
                       "position_in_order": r["position_in_order"], "solves": r["solves"], "window_s": r["window_s"],
                       "seconds_per_solve": r["seconds_per_solve"], "eq_tops": r["eq_tops"], "watts": w,
                       "energy_j": r["energy_j"], "joule_per_solve": r["joule_per_solve"],
                       "eq_tops_per_watt": r["eq_tops"] / w if w else None, "temp_max_c": r["temp_max_c"],
                       "sm_mhz_median": r["sm_mhz_median"], "mem_mhz_median": r["mem_mhz_median"],
                       "clock_event_reasons_or": r["clock_event_reasons_or"], "energy_source": r["energy_source"],
                       "verified": m["verified"]}
        pr = run["parallel"]["repetitions"]
        for r in pr:
            yield {"file": path, "label": d["label"], "device_type": "gpu", "gpu": g["name"], "machine": d["system"]["machine"],
                   "driver": g["driver"], "cuda_runtime": g["cuda_runtime"], "power_limit_w": g["power_limit_w"],
                   "tool_version": t["version"], "git_commit": t["git_commit"], "n": run["n"], "p": par["p"],
                   "seed": par["seed"], "method": "P_all_four_parallel", "repeat": r["repeat"], "position_in_order": 5,
                   "solves": sum(r["solves"]), "window_s": r["window_s"], "seconds_per_solve": None,
                   "eq_tops": r["eq_tops"], "watts": r["watts"], "energy_j": r["energy_j"], "joule_per_solve": None,
                   "eq_tops_per_watt": r["eq_tops"] / r["watts"] if r["watts"] else None,
                   "temp_max_c": r["temp_max_c"], "sm_mhz_median": r["sm_mhz_median"], "mem_mhz_median": None,
                   "clock_event_reasons_or": None, "energy_source": None, "verified": d["all_verified"]}


def ci95(v):
    if len(v) < 2:
        return float("nan")
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045}
    df = len(v) - 1
    q = t.get(df) or (2.0 if df <= 60 else 1.96)
    return q * statistics.stdev(v) / math.sqrt(len(v))


def welch(a, b):
    """Welch t statistic, Welch-Satterthwaite df and two-sided p-value (Student t via regularized beta)."""
    ma, mb, va, vb = statistics.mean(a), statistics.mean(b), statistics.variance(a), statistics.variance(b)
    se2 = va / len(a) + vb / len(b)
    if se2 == 0:
        return float("inf"), float("nan"), 0.0
    t = (mb - ma) / math.sqrt(se2)
    df = se2 ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    return t, df, betainc(df / 2, 0.5, df / (df + t * t))


def betainc(a, b, x):  # regularized incomplete beta I_x(a,b), continued fraction (Numerical Recipes)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x > (a + 1) / (a + b + 2):
        return 1 - betainc(b, a, 1 - x)
    c, d, f = 1.0, 1 - (a + b) * x / (a + 1), 0.0
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    f = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + num * d
            d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + num / c if abs(1 + num / c) > 1e-300 else 1e-300
            f *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return math.exp(lbt) * f / a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", help="write tidy CSV to this file")
    ap.add_argument("--compare", action="store_true", help="compare exactly two files (second / first)")
    a = ap.parse_args()
    data = [r for f in a.files for r in rows(f)]
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {len(data)} rows to {a.csv}")

    groups = {}
    for r in data:
        groups.setdefault((r["file"], r["gpu"], r["n"], r["method"]), []).append(r)
    print(f"{'gpu':<18} {'n':>6} {'method':<30} {'k':>3} {'eq.TOPS':>16} {'W':>7} {'TOPS/W':>7} {'MHz':>5} {'Tmax':>4}")
    for (f, gpu, n, m), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3])):
        v = [x["eq_tops"] for x in g]
        print(f"{gpu[:18]:<18} {n:>6} {m:<30} {len(v):>3} {statistics.mean(v):8.2f} +-{ci95(v):6.2f} "
              f"{mean(x['watts'] for x in g):7.1f} {mean(x['eq_tops_per_watt'] for x in g):7.3f} "
              f"{statistics.median(x['sm_mhz_median'] for x in g):5.0f} {max(x['temp_max_c'] or 0 for x in g):4}")

    if a.compare:
        if len(a.files) != 2:
            sys.exit("--compare needs exactly two files")
        fa, fb = a.files
        print(f"\nratio {fb} / {fa}  (Welch t-test on eq.TOPS per repetition)")
        print(f"{'n':>6} {'method':<30} {'ratio TOPS':>10} {'ratio TOPS/W':>12} {'t':>8} {'df':>6} {'p':>9}")
        for (f, gpu, n, m), g in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][3])):
            if f != fa:
                continue
            other = [v for k, v in groups.items() if k[0] == fb and k[2] == n and k[3] == m]
            if not other or len(g) < 2 or len(other[0]) < 2:
                continue
            x, y = [r["eq_tops"] for r in g], [r["eq_tops"] for r in other[0]]
            ex = mean(r["eq_tops_per_watt"] for r in g)
            ey = mean(r["eq_tops_per_watt"] for r in other[0])
            t, df, p = welch(x, y)
            print(f"{n:>6} {m:<30} {statistics.mean(y) / statistics.mean(x):10.3f} {ey / ex:12.3f} {t:8.2f} {df:6.1f} {p:9.2g}")


if __name__ == "__main__":
    main()
