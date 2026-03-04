#!/usr/bin/env bash
set -e

git ls-files '*.cpp' '*.h' \
  | xargs wc -l \
  | tail -n 1