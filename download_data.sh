#!/usr/bin/env bash
# Downloads the BFCL v3 files used here (Apache-2.0, gorilla-llm/Berkeley-Function-Calling-Leaderboard).
set -e
cd "$(dirname "$0")" && mkdir -p data && cd data
B=https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main
for f in BFCL_v3_simple BFCL_v3_multiple BFCL_v3_irrelevance BFCL_v3_live_simple BFCL_v3_live_multiple; do
  curl -sL -o "$f.json" "$B/$f.json"
done
for f in BFCL_v3_simple BFCL_v3_multiple BFCL_v3_live_simple BFCL_v3_live_multiple; do
  curl -sL -o "ans_$f.json" "$B/possible_answer/$f.json"
done
