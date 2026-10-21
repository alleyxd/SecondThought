"""Collects cheap gating signals for every item: one no-think call with token logprobs, and a one-token probe asking
whether any tool fits. Appends to results/signals-<label>.jsonl (resumable)."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path

from openai import AsyncOpenAI

from .bfcl import CATEGORIES, load, safe_name, score, to_tool

PROBE = ("\n\nBefore answering: can this request be fulfilled by calling one of the available tools? "
         "Answer with a single letter, A for yes or B for no.")


def first_token_p(entry, want: str) -> float:
    for alt in entry.top_logprobs or []:
        if alt.token.strip() == want:
            return math.exp(alt.logprob)
    return 0.0


async def one(client, item):
    tools = [to_tool(f) for f in item["function"]]
    names = {safe_name(f["name"]): f["name"] for f in item["function"]}
    messages = [dict(m) for turn in item["question"] for m in turn]
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    r = await client.chat.completions.create(model="local", messages=messages, tools=tools, temperature=0.0,
                                             max_tokens=1024, logprobs=True, top_logprobs=5, extra_body=off)
    ch = r.choices[0]
    calls = []
    for tc in ch.message.tool_calls or []:
        try:
            calls.append((tc.function.name, json.loads(tc.function.arguments or "{}")))
        except json.JSONDecodeError:
            calls.append((tc.function.name, None))
    lps = [t.logprob for t in (ch.logprobs.content or [])] if ch.logprobs else []
    p_call = first_token_p(ch.logprobs.content[0], "<tool_call>") if lps else 0.0
    probe_msgs = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + PROBE}]
    pr = await client.chat.completions.create(model="local", messages=probe_msgs, tools=tools, tool_choice="none",
                                              temperature=0.0, max_tokens=1, logprobs=True, top_logprobs=10,
                                              extra_body=off)
    e = pr.choices[0].logprobs.content[0]
    pa, pb = first_token_p(e, "A"), first_token_p(e, "B")
    return {"id": item["id"], "category": item["category"],
            "score": score(item, [(n, a) for n, a in calls if a is not None], names), "n_calls": len(calls),
            "p_call": p_call, "min_lp": min(lps) if lps else 0.0, "mean_lp": sum(lps) / len(lps) if lps else 0.0,
            "n_tokens": len(lps), "p_probe_yes": pa / (pa + pb) if pa + pb > 0 else 0.5}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--duty", type=float, default=0.5)
    args = p.parse_args()
    items = []
    for cat in CATEGORIES:
        n = 300 if cat == "live_multiple" else None
        items += [it for it in load(cat, n) if cat == "irrelevance" or it["ground_truth"]]
    out = Path("results") / f"signals-{args.label}.jsonl"
    done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="local", timeout=600)
    print(f"{args.label}: {len(done)} done, {len(todo)} to go", flush=True)
    import time
    with out.open("a", encoding="utf-8") as f:
        for s in range(0, len(todo), args.batch):
            t0 = time.perf_counter()
            res = await asyncio.gather(*(one(client, it) for it in todo[s:s + args.batch]))
            busy = time.perf_counter() - t0
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"{args.label}: {len(done) + s + len(res)}/{len(items)}", flush=True)
            await asyncio.sleep(busy * (1 - args.duty) / args.duty)
    print(f"{args.label}: finished", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
