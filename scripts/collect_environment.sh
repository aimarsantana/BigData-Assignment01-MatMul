#!/usr/bin/env bash
# Record the hardware/software environment of the benchmark machine.
cd "$(dirname "$0")/.."
out=results/environment.txt
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "--- lscpu ---"; lscpu
  echo "--- memory ---"; free -b
  echo "--- os ---"; cat /etc/os-release; uname -a
  echo "--- gcc ---"; gcc --version | head -1
  echo "--- java ---"; java -version 2>&1 | grep -v JAVA_TOOL_OPTIONS
  echo "--- javac ---"; javac -version 2>&1 | grep -v JAVA_TOOL_OPTIONS
  echo "--- python ---"; python3 --version
  python3 -c "import numpy, pandas, matplotlib; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('matplotlib', matplotlib.__version__)"
  echo "--- cpu governor / scaling (may be unavailable in VMs) ---"
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "not exposed"
} > "$out" 2>&1
echo "wrote $out"
