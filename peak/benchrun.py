#!/usr/bin/env python3
"""Benchmark runner with GPU telemetry: per test throughput + temperature, power draw, clock, PCIe link.
Usage: benchrun.py [benchmark command ...]   (default ./gpubench). Minimum duration per test: GPUBENCH_MIN_S (default 6)."""
import csv, os, subprocess, sys, time, statistics as st

cmd = sys.argv[1:] or ["./gpubench"]
os.environ.setdefault("GPUBENCH_MIN_S", "6")
Q = "timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active,fan.speed"
log = "/tmp/gpubench-telemetry.csv"
mon = subprocess.Popen(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits", "-lms", "200", "-f", log])
time.sleep(2)
idle_end = time.time()
out = subprocess.run(cmd, capture_output=True, text=True).stdout
time.sleep(1); mon.terminate(); mon.wait()

rows = []
for r in csv.reader(open(log)):
    r = [x.strip() for x in r]
    try:
        t = time.mktime(time.strptime(r[0].split(".")[0], "%Y/%m/%d %H:%M:%S")) + float("0." + r[0].split(".")[1])
        rows.append((t, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), r[6], r[7], r[8], r[9]))
    except (ValueError, IndexError):
        pass

print("\n".join(l for l in out.splitlines() if not l.startswith("RESULT")))
idle = [x for x in rows if x[0] < idle_end]
if idle:
    print(f"idle       {st.median(x[1] for x in idle):.0f} C, {st.median(x[2] for x in idle):.1f} W")
print(f"{'test':20s} {'value':>9s} {'T max':>6s} {'P Med':>6s} {'P max':>6s} {'SM MHz':>7s} {'Util':>5s} {'PCIe':>6s} {'value/W':>8s}  throttle")
for l in out.splitlines():
    if not l.startswith("RESULT"):
        continue
    _, k, v, win = l.split()
    a, b = map(float, win[2:].split("-"))
    s = [x for x in rows if a + 0.5 <= x[0] <= b]  # hide ramp-up
    if not s:
        print(f"{k:20s} {float(v):9.2f}  (no telemetry)"); continue
    p = st.median(x[2] for x in s)
    link = max({f"G{x[6]}x{x[7]}" for x in s})
    reasons = sorted({x[8] for x in s} - {"0x0000000000000000", "0x0000000000000001"})
    print(f"{k:20s} {float(v):9.2f} {max(x[1] for x in s):6.0f} {p:6.1f} {max(x[2] for x in s):6.1f} "
          f"{st.median(x[3] for x in s):7.0f} {st.median(x[5] for x in s):5.0f} {link:>6s} {float(v)/p:8.3f}  {','.join(reasons) or '-'}")
