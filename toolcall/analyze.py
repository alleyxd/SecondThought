"""Tables for the paper: accuracy per model and condition on tool-needing items, irrelevance accuracy, paired
difference to base with a bootstrap 95% interval, tokens, latency and retry rate."""
import json
import random
from collections import defaultdict
from pathlib import Path

MODELS = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
CONDS = ["base", "think", "no_desc", "no_desc_all", "opaque_names", "distract8", "distract32", "retrieve5",
         "retrieve10", "dense5", "required", "repair"]


def ci(diffs, iters=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return sum(diffs) / len(diffs), means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    summary = []
    for m in MODELS:
        rows = [json.loads(x) for x in Path(f"results/{m}.jsonl").read_text(encoding="utf-8").splitlines() if x]
        by = defaultdict(dict)
        for r in rows:
            by[r["cond"]][r["id"]] = r
        base = by["base"]
        print(f"\n{m}")
        print(f"  {'cond':13} {'tool acc':>8} {'irrel':>6} {'vs base [95% CI]':>26} {'tokens':>7} {'secs':>6} {'retry':>6}")
        for c in CONDS:
            cur = by[c]
            tool = [r for r in cur.values() if r["category"] != "irrelevance"]
            irr = [r for r in cur.values() if r["category"] == "irrelevance"]
            acc = sum(r["score"] for r in tool) / len(tool)
            iacc = sum(r["score"] for r in irr) / len(irr)
            d = ci([cur[i]["score"] - base[i]["score"] for i in cur if base[i]["category"] != "irrelevance"])
            tok = sum(r.get("tokens", 0) for r in cur.values()) / len(cur)
            secs = sorted(r.get("secs", 0) for r in cur.values())[len(cur) // 2]
            retry = sum(r.get("retried", False) for r in cur.values()) / len(cur)
            print(f"  {c:13} {acc:8.3f} {iacc:6.3f} {d[0]:+8.3f} [{d[1]:+.3f}, {d[2]:+.3f}] {tok:7.0f} {secs:6.2f} {retry:6.3f}")
            summary.append({"model": m, "cond": c, "tool_acc": acc, "irrelevance_acc": iacc, "diff": d[0],
                            "lo": d[1], "hi": d[2], "tokens": tok, "median_secs": secs, "retry_rate": retry})
    Path("results/summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
