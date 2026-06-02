#!/usr/bin/env bash
# 用法: ./scripts/configure.sh [debug|release] [extra cmake args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-debug}"
shift || true

case "$MODE" in
  debug)
    BUILD_DIR="$ROOT/build"
    BUILD_TYPE=Debug
    ;;
  release)
    BUILD_DIR="$ROOT/build-release"
    BUILD_TYPE=Release
    ;;
  *)
    echo "用法: $0 [debug|release] [cmake 选项...]" >&2
    exit 1
    ;;
esac

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  "$@"

echo "Configured: $BUILD_DIR ($BUILD_TYPE)"
