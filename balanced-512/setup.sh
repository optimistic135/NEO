#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pin=73c7669984cf57e58d943d2649c2ad1e240d9b01
mkdir -p .runtime
git -C .. archive "$pin" paper-exp | tar -x -C .runtime
