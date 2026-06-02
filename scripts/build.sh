#!/usr/bin/env bash
# 用法: ./scripts/build.sh [debug|release] [target] [-j N]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-debug}"
TARGET="${2:-all_cuda}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -ge 1 ]]; then shift; fi

case "$MODE" in
  debug) BUILD_DIR="$ROOT/build" ;;
  release) BUILD_DIR="$ROOT/build-release" ;;
  *)
    echo "用法: $0 [debug|release] [target] [cmake --build 额外参数]" >&2
    exit 1
    ;;
esac

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  "$ROOT/scripts/configure.sh" "$MODE"
fi

cmake --build "$BUILD_DIR" --target "$TARGET" -j "${CUDAPRO_JOBS:-$(nproc)}" "$@"
echo "Binaries: $BUILD_DIR/bin/"
