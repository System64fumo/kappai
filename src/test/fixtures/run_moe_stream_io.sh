#!/usr/bin/env bash
# Standalone MoE fault-injection regression, run sequentially (single fixture pathname).
set -euo pipefail
KAPPAI_SRC=${KAPPAI_SRC:-$(cd "$(dirname "$0")/../../../" && pwd)}
: "${KAPPAI_BUILD:?set KAPPAI_BUILD to the built output directory}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export KAPPAI_IO_FIXTURE="$TMP/experts.bin"
cc -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"$KAPPAI_SRC/src" "$KAPPAI_SRC/src/test/fixtures/test_moe_stream_io.c" \
  -L"$KAPPAI_BUILD" -Wl,-rpath,"$KAPPAI_BUILD" \
  -lkappai -lpthread -lm -o "$TMP/verify_io"
export ASAN_OPTIONS=detect_leaks=0
for api in resolve prep; do
  for mode in bounce aligned buffered; do
    for fault in healthy eio eio-once eof eintr; do
      "$TMP/verify_io" "$mode" "$api" "$fault" nommap
    done
  done
  for mode in bounce aligned; do
    "$TMP/verify_io" "$mode" "$api" direct-eio nommap
  done
  for fault in short partial-eio; do
    "$TMP/verify_io" buffered "$api" "$fault" nommap
  done
  for mode in bounce aligned; do
    for fault in healthy eio eio-once direct-eio eintr; do
      "$TMP/verify_io" "$mode" "$api" "$fault" mmap
    done
  done
done
