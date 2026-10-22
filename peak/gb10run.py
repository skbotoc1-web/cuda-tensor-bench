#!/usr/bin/env python3
"""gb10run.py - runs a benchmark and samples nvidia-smi every 500 ms in parallel (works on any NVIDIA GPU).
Assigns median power, max. temperature and median SM clock to every 'RESULT name value t=a-b' line.
GB10: no memory clock / power-limit sensors; power.draw is GPU power as reported by the driver (CPU share not included)."""
import subprocess, sys, threading, time, re, statistics as st

samples, stop = [], False
def sampler():
    while not stop:
        t = time.time()
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            v = [float(x) for x in o.strip().split(",")]
            samples.append((t, *v))
        except Exception:
            pass
        time.sleep(max(0, 0.5 - (time.time() - t)))

idle = []
for _ in range(6):
    o = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    idle.append([float(x) for x in o.split(",")]); time.sleep(0.5)
th = threading.Thread(target=sampler, daemon=True); th.start()
p = subprocess.run(sys.argv[1:], capture_output=True, text=True)
stop = True; th.join()
print(p.stdout, end=""); print(p.stderr, end="", file=sys.stderr)
print(f"idle {st.median(i[0] for i in idle):.0f} C, {st.median(i[1] for i in idle):.1f} W")
print(f"{'test':<18}{'value':>9}{'T max':>7}{'P Med':>7}{'P max':>7}{'MHz':>6}{'Util':>5}{'value/W':>8}{'J/Op':>8}  n")
for m in re.finditer(r"RESULT (\S+) ([\d.]+) t=([\d.]+)-([\d.]+)", p.stdout):
    name, val, a, b = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
    s = [x for x in samples if a + 1.0 <= x[0] <= b]          # drop the first second (ramp-up)
    if not s:
        print(f"{name:<18}{val:9.2f}  (no samples)"); continue
    pm = st.median(x[2] for x in s)
    n = int(re.search(r"_(\d+)$", name).group(1)); flop = 2.0 * n ** 3
    print(f"{name:<18}{val:9.2f}{max(x[1] for x in s):7.0f}{pm:7.1f}{max(x[2] for x in s):7.1f}"
          f"{st.median(x[3] for x in s):6.0f}{st.median(x[4] for x in s):5.0f}{val / pm:8.3f}{pm * flop / (val * 1e12):8.3f}  {len(s)}")
sys.exit(p.returncode)
