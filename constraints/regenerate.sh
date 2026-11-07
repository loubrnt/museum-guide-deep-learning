#!/usr/bin/env bash
# Regenerates constraints/py311.txt: the versions the test suite passes with, on Python 3.11 / Linux (the CI setup).
#
#   bash constraints/regenerate.sh          # needs Docker; takes a few minutes (downloads PyTorch)
#
# It installs the project (`pip install -e ".[dev]"`, CPU-only PyTorch like the CI) in a clean python:3.11-slim
# container, RUNS THE TEST SUITE there, and only if the tests pass writes the frozen versions.
# torch and torchvision are not pinned exactly (their Linux wheels carry a "+cpu" tag and differ from the macOS
# ones): the file keeps the minor version that was tested as a range.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

docker run --rm -v "$PWD":/src:ro -v "$OUT":/out python:3.11-slim bash -ec '
  export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
  mkdir /work && tar -C /src --exclude=./data --exclude=./.git -cf - . | tar -C /work -xf -
  cd /work
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -q -e ".[dev]"
  python -m pytest -q -p no:cacheprovider | tail -3
  pip freeze --exclude-editable > /out/freeze.txt
  python -c "import platform, sys; print(sys.version.split()[0], platform.machine())" > /out/python.txt
'

python3 - "$OUT" > constraints/py311.txt <<'PY'
import sys, datetime, pathlib
out = pathlib.Path(sys.argv[1])
py, arch = (out / "python.txt").read_text().split()
pins = dict(line.strip().split("==", 1) for line in (out / "freeze.txt").read_text().splitlines() if "==" in line)
torch, vision = pins.pop("torch").split("+")[0], pins.pop("torchvision").split("+")[0]
minor = lambda v: ".".join(v.split(".")[:2])
def upper(v):  # next minor version
    major, mnr = (int(x) for x in v.split(".")[:2]); return f"{major}.{mnr + 1}"
print(f"# Known-good versions: the test suite passes with these (Python {py}, Linux {arch}, CPU-only PyTorch).")
print(f"# Generated on {datetime.date.today()} by constraints/regenerate.sh. Use: pip install -c constraints/py311.txt -e \".[dev]\"")
print("# Optional extras (multilingual, demo) are not covered. torch/torchvision are ranges, not exact pins: their wheels")
print("# differ between platforms (Linux CPU wheels carry a +cpu tag).")
print(f"torch>={minor(torch)},<{upper(torch)}")
print(f"torchvision>={minor(vision)},<{upper(vision)}")
for name in sorted(pins, key=str.lower):
    print(f"{name}=={pins[name]}")
PY
echo "wrote constraints/py311.txt ($(grep -c '==' constraints/py311.txt) pins)"
