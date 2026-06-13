#!/usr/bin/env bash
# scripts/race/run_all.sh
#
# Thin wrapper around the CMake race target. Dependency tracking lives in
# cmake/Race.cmake — `make` only reruns phases whose inputs changed.
#
#   cmake -B build .
#   cmake --build build --target race

#
# Skip Python phases:
#   cmake -B build -DOLMO_RACE_SKIP_PYTHON=ON .
#   cmake --build build --target race

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
BUILD="${BUILD_DIR:-build}"

if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
  printf "\033[1;36m[run_all]\033[0m configuring → %s\n" "$BUILD"
  cmake -B "$BUILD" .
fi

printf "\033[1;36m[run_all]\033[0m building race target (incremental)\n"
cmake --build "$BUILD" --target race -j8

printf "\n\033[1;32m✓  RACE COMPLETE\033[0m\n"
printf "Read:  %s/scripts/race/results/RESULT.md\n" "$ROOT"
