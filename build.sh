#!/usr/bin/env bash
# Build the C and Java implementations with the exact flags used in the study.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

CFLAGS="-O2 -std=c11 -Wall -Wextra"
gcc $CFLAGS -DCFLAGS_STR="\"$CFLAGS\"" -o build/matmul src/c/matmul.c
echo "built build/matmul with: gcc $CFLAGS"

javac --release 21 -d build src/java/MatMul.java
echo "built build/MatMul.class with: javac --release 21"
