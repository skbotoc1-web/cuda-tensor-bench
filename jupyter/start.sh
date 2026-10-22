#!/bin/sh
# Start JupyterLab in /work. The analysis notebook is copied there once and never overwritten,
# so your own edits and outputs stay in the mounted folder.
set -e
mkdir -p /work/results /work/export
cp -n /opt/trianglebench/notebooks/*.ipynb /work/ 2>/dev/null || true
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.root_dir=/work "$@"
