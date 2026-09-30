#!/usr/bin/env bash
# Build and run the offline personalization / A-B-selection self-test.
# Proves the GazeNet training, A/B model selection and profile persistence are
# numerically sound without needing a camera or a display.
set -euo pipefail
cd "$(dirname "$0")"
S=Sources/Paralic
OUT="${TMPDIR:-/tmp}/paralic-selftest"
echo "Compiling self-test…"
swiftc -O \
  "$S/ML/Linalg.swift" "$S/ML/GazeNet.swift" "$S/ML/ABSelection.swift" \
  "$S/Tools/main.swift" \
  -framework Accelerate -o "$OUT"
echo "Running…"; echo
exec "$OUT"
