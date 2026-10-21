"""Confidence-gated thinking and abstention, evaluated offline from the direct-call signals and the stored think runs.

Policy for each request, using only the normal no-think call:
    p_call < t_abstain          -> decline (no tool call)
    mean token logprob >= t_think -> keep the direct call
    otherwise                   -> rerun with thinking and use that answer
Thresholds are picked by 5-fold cross-validation to maximise accuracy over all items (tool and irrelevance).
Cost is output tokens: the direct call always, plus the think call when triggered.
"""
import json
import random
from pathlib import Path

MODELS = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"]


def load(m):
    sig = {r["id"]: r for r in map(json.loads, Path(f"results/signals-{m}.jsonl").read_text().splitlines())}
    runs = {}
    for line in Path(f"results/{m}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["cond"] in ("base", "think"):
            runs[(r["id"], r["cond"])] = r
    items = []
    for i, s in sig.items():
        th = runs[(i, "think")]
        items.append({"irr": s["category"] == "irrelevance", "s0": s["score"], "s1": th["score"],
                      "t0": s["n_tokens"], "t1": th["tokens"], "p_call": s["p_call"], "conf": s["mean_lp"],
                      "called": s["n_calls"] > 0})
    return items


def evaluate(items, ta, tc):
    acc = cost = thinks = 0
    for it in items:
        cost += it["t0"]
        if it["p_call"] < ta:
            acc += it["irr"]
        elif it["conf"] >= tc:
            acc += it["s0"]
        else:
            acc += it["s1"]
            cost += it["t1"]
            thinks += 1
    n = len(items)
    return acc / n, cost / n, thinks / n


def grid(items):
    qs = sorted(it["conf"] for it in items)
    tcs = [qs[int(q * (len(qs) - 1))] for q in [i / 20 for i in range(21)]] + [1e9]
    tas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    return tas, tcs


def cv(items, k=5, seed=0):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    tot = {"acc": 0, "cost": 0, "think": 0}
    for f in folds:
        test = [items[i] for i in f]
        train = [items[i] for i in idx if i not in set(f)]
        tas, tcs = grid(train)
        ta, tc = max(((a, c) for a in tas for c in tcs), key=lambda p: evaluate(train, *p)[0])
        a, c, t = evaluate(test, ta, tc)
        for key, v in zip(tot, (a, c, t)):
            tot[key] += v * len(test)
    return {key: v / len(items) for key, v in tot.items()}


def random_gate(items, rate, ta, seed=0):
    rng = random.Random(seed)
    acc = cost = 0
    for it in items:
        cost += it["t0"]
        if it["p_call"] < ta:
            acc += it["irr"]
        elif rng.random() < rate:
            acc += it["s1"]
            cost += it["t1"]
        else:
            acc += it["s0"]
    return acc / len(items), cost / len(items)


def auroc(pos, neg):
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    s = sum(i for i, (_, lab) in enumerate(ranked, 1) if lab)
    return (s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def signals_table():
    print("AUROC: predicting a correct direct call (tool items) / predicting that a tool is needed (all items)")
    for m in MODELS:
        sig = [json.loads(x) for x in Path(f"results/signals-{m}.jsonl").read_text().splitlines()]
        tool = [r for r in sig if r["category"] != "irrelevance"]
        row = []
        for f in ["mean_lp", "min_lp", "p_call", "p_probe_yes"]:
            a = auroc([r[f] for r in tool if r["score"] == 1], [r[f] for r in tool if r["score"] == 0])
            b = auroc([r[f] for r in sig if r["category"] != "irrelevance"],
                      [r[f] for r in sig if r["category"] == "irrelevance"])
            row.append(f"{f} {a:.2f}/{b:.2f}")
        print(f"  {m}: " + " | ".join(row))


def budget_curve(rates=(0.1, 0.2, 0.3, 0.4)):
    """Think only on direct calls with the lowest mean token logprob, up to a budget; declined requests stay declined."""
    print("Accuracy at a thinking budget (share of all requests), confidence-ranked vs random, output tokens")
    for m in MODELS:
        items = load(m)
        n = len(items)
        cand = sorted([i for i in items if i["called"]], key=lambda i: i["conf"])
        line = f"  {m}: direct {sum(i['s0'] for i in items) / n:.3f}, think {sum(i['s1'] for i in items) / n:.3f} "
        line += f"({sum(i['t0'] + i['t1'] for i in items) / n:.0f} tok incl. direct pass) |"
        for rate in rates:
            k = int(rate * n)
            for name, pick in (("conf", cand[:k]), ("rand", random.Random(0).sample(cand, min(k, len(cand))))):
                chosen = set(map(id, pick))
                acc = sum(i["s1"] if id(i) in chosen else i["s0"] for i in items) / n
                if name == "conf":
                    tok = sum(i["t0"] + (i["t1"] if id(i) in chosen else 0) for i in items) / n
                    line += f" {int(rate * 100)}%: {acc:.3f} ({tok:.0f} tok)"
                else:
                    line += f" rand {acc:.3f};"
        print(line)


def main():
    signals_table()
    budget_curve()
    for m in MODELS:
        items = load(m)
        n = len(items)
        direct = sum(it["s0"] for it in items) / n, sum(it["t0"] for it in items) / n
        think = sum(it["s1"] for it in items) / n, sum(it["t1"] for it in items) / n
        abstain_only = cv([{**it, "conf": 1e9} for it in items])
        gated = cv(items)
        rnd = random_gate(items, gated["think"], 0.05)
        oracle = sum(max(it["s0"], it["s1"]) for it in items) / n
        print(f"{m}: direct {direct[0]:.3f} ({direct[1]:.0f} tok) | think {think[0]:.3f} ({think[1]:.0f} tok) | "
              f"abstain-gate {abstain_only['acc']:.3f} | gated {gated['acc']:.3f} ({gated['cost']:.0f} tok, "
              f"thinks {gated['think']:.0%}) | random at same rate {rnd[0]:.3f} | oracle {oracle:.3f}")


if __name__ == "__main__":
    main()
