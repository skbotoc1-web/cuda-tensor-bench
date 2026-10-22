#!/usr/bin/env python3
"""cutlasspeak.py - INT4 dense and 2:4-sparse GEMM peak via cutlass_profiler.
Stage 1: every built kernel of a class is profiled briefly (with CUTLASS's own verification against a reference).
Stage 2: the fastest kernel runs >= GPUBENCH_MIN_S seconds; a RESULT line with the time window is printed for
benchrun.py telemetry. Sparse throughput is reported as cutlass_profiler does: dense-equivalent (effective) 2*m*n*k.
Usage: cutlasspeak.py [class,...] [n,...]   classes: int4, sp_int4, sp_int8, sp_fp16, int8
"""
import csv, glob, io, os, subprocess, sys, time

PROF = os.environ.get("CUTLASS_PROFILER", "cutlass_profiler")  # path to a cutlass_profiler build
MIN_S = float(os.environ.get("GPUBENCH_MIN_S", "8"))
CLASSES = {  # kernel-name filter, operation, unit
    "int4":    ("cutlass_tensorop_s4_i16864gemm_s4_*_tn_*,cutlass_tensorop_i16864gemm_s4_*_tn_*", "gemm", "TOPS"),
    "int8":    ("cutlass_tensorop_s8_i16832gemm_s8_*_tn_*,cutlass_tensorop_i16832gemm_s8_*_tn_*", "gemm", "TOPS"),
    "sp_int4": ("*i168128spgemm_s4*", "spgemm", "TOPS"),
    "sp_int8": ("*i16864spgemm_s8*", "spgemm", "TOPS"),
    "sp_fp16": ("*s16832spgemm_f16*,*h16832spgemm*", "spgemm", "TFLOPS"),
}
classes = (sys.argv[1] if len(sys.argv) > 1 else "int8,int4,sp_int8,sp_int4,sp_fp16").split(",")
sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8192").split(",")]
tmp = os.environ.get("TMPDIR", "/tmp")


def profile(kernels, op, n, extra):
    out = f"{tmp}/cutlasspeak_{os.getpid()}"
    for f in glob.glob(out + "*.csv"):
        os.remove(f)
    cmd = [PROF, f"--operation={op}", f"--kernels={kernels}", f"--m={n}", f"--n={n}", f"--k={n}",
           "--providers=cutlass", f"--output={out}", "--report-not-run=true"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    rows = []
    for f in glob.glob(out + "*.csv"):
        rows += list(csv.DictReader(open(f)))
    return p.returncode, rows, p.stdout + p.stderr


print(f"{'class':8s} {'n':>6s} {'cand.':>5s} {'best':>9s} {'Verify':>8s}  Kernel")
rc = 0
for c in classes:
    kern, op, unit = CLASSES[c]
    for n in sizes:
        code, rows, log = profile(kern, op, n, ["--verification-enabled=true", "--verification-providers=device",
                                                "--warmup-iterations=5", "--profiling-iterations=30"])
        ok = [r for r in rows if r.get("Status") == "success" and r.get("Disposition") in ("passed", "not_verified")]
        if not ok:
            print(f"{c:8s} {n:6d}  no runnable kernel (rc={code}, rows={len(rows)})")
            if rows:
                print("   ", {(r.get('Status'), r.get('Disposition')) for r in rows})
            else:
                print("   ", log.strip().splitlines()[-3:] if log.strip() else "-")
            rc = 2
            continue
        failed = [r for r in rows if r.get("Disposition") in ("failed", "incorrect")]
        best = max(ok, key=lambda r: float(r["GFLOPs"]))
        disp = best["Disposition"]
        print(f"{c:8s} {n:6d} {len(ok):5d} {float(best['GFLOPs'])/1000:9.2f} {disp:>8s}  {best['Operation']}"
              + (f"  ({len(failed)} kernels FAILED verification)" if failed else ""), flush=True)
        # cutlass_profiler has no reference provider for spgemm -> "not_verified"; sparse correctness is shown by
        # sparsepeak (cuSPARSELt, sampled double-precision check). Dense classes must pass verification.
        if failed or (disp != "passed" and op != "spgemm"):
            rc = 2
        # stage 2: sustained run of the best kernel, no verification, duration-based
        # one invocation (no restarts = no idle gaps in the telemetry window); value = mean over the whole duration
        t0 = time.time()
        code, r2, _ = profile(best["Operation"], op, n, ["--verification-enabled=false", "--warmup-iterations=50", "--profiling-iterations=0",
                                                           f"--profiling-duration={int(MIN_S * 1000)}", "--min-iterations=20"])
        vals = [float(r["GFLOPs"]) / 1000 for r in r2 if r.get("Status") == "success"]
        if not vals:
            print(f"{c:8s} {n:6d}  Dauerlauf fehlgeschlagen (rc={code})"); rc = 2; continue
        print(f"RESULT {c}_{n} {vals[0]:.2f} t={t0 + 1.0:.1f}-{time.time():.1f}", flush=True)
print("FAIL cutlasspeak" if rc else "OK cutlasspeak done")
sys.exit(rc)
