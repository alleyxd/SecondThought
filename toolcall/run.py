"""Runs BFCL categories under each tool-calling condition against an OpenAI-compatible server (vLLM with
--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3) and appends to a resumable journal.

Conditions
    base            tools as given, thinking off, tool_choice=auto
    think           same with thinking on
    no_desc         function descriptions removed (parameter descriptions kept)
    no_desc_all     function and parameter descriptions removed
    opaque_names    functions renamed tool_1..tool_k in random order
    distract8/32    8 or 32 unrelated tools from other examples added, order shuffled
    retrieve5       32 distractors added, then only the 5 tools ranked highest by BM25 on the question are sent
    retrieve10      the same with the top 10
    dense5          the same with the top 5 by bge-small-en-v1.5 cosine similarity
    required        tool_choice=required (the server must emit a call)
    repair          base, then one retry with the validation errors when a call breaks the schema or names an
                    unknown tool
"""
import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

from .bfcl import CATEGORIES, load, safe_name, score, to_tool, validate

CONDITIONS = ["base", "think", "no_desc", "no_desc_all", "opaque_names", "distract8", "distract32", "retrieve5",
              "required", "repair", "retrieve10", "dense5"]
_EMBED = None


def dense_top(query: str, docs: list[str], k: int) -> list[int]:
    global _EMBED
    if _EMBED is None:
        from sentence_transformers import SentenceTransformer
        _EMBED = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    q = _EMBED.encode([query], normalize_embeddings=True)
    d = _EMBED.encode(docs, normalize_embeddings=True)
    sims = (d @ q.T).ravel()
    return sorted(range(len(docs)), key=lambda i: -sims[i])[:k]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_top(query: str, docs: list[str], k: int) -> list[int]:
    toks = [tokens(d) for d in docs]
    avg = sum(map(len, toks)) / max(len(toks), 1)
    df = Counter(t for d in toks for t in set(d))
    n = len(docs)
    q = tokens(query)
    scores = []
    for i, d in enumerate(toks):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])[:k]


def build(item: dict, cond: str, pool: list[dict], rng: random.Random):
    """Returns (tools, names: sent name -> original name, extra request kwargs)."""
    fns = list(item["function"])
    own = {f["name"] for f in fns}
    kw = {}
    if cond in ("distract8", "distract32", "retrieve5", "retrieve10", "dense5"):
        k = 8 if cond == "distract8" else 32
        extra = [f for f in rng.sample(pool, min(len(pool), k * 3)) if f["name"] not in own]
        seen = set(own)
        for f in extra:
            if len(fns) - len(item["function"]) >= k:
                break
            if safe_name(f["name"]) not in {safe_name(x) for x in seen}:
                fns.append(f)
                seen.add(f["name"])
        rng.shuffle(fns)
        if cond in ("retrieve5", "retrieve10", "dense5"):
            query = " ".join(m["content"] for turn in item["question"] for m in turn if m["role"] == "user")
            docs = [f"{f['name']} {f.get('description', '')}" for f in fns]
            top = dense_top(query, docs, 5) if cond == "dense5" else bm25_top(query, docs, 10 if cond == "retrieve10" else 5)
            fns = [fns[i] for i in top]
    if cond == "opaque_names":
        order = list(range(len(fns)))
        rng.shuffle(order)
        tools, names = [], {}
        for j, idx in enumerate(order, 1):
            tools.append(to_tool(fns[idx], name=f"tool_{j}"))
            names[f"tool_{j}"] = fns[idx]["name"]
        return tools, names, kw
    tools = [to_tool(f, describe=cond not in ("no_desc", "no_desc_all"), describe_params=cond != "no_desc_all")
             for f in fns]
    names = {safe_name(f["name"]): f["name"] for f in fns}
    if cond == "required":
        kw["tool_choice"] = "required"
    return tools, names, kw


def parse_calls(msg) -> tuple[list[tuple[str, dict]], list[str]]:
    calls, problems = [], []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = None
            problems.append(f"arguments for {tc.function.name} are not valid JSON")
        calls.append((tc.function.name, args))
    return calls, problems


async def ask(client, model, messages, tools, think, kw):
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model, messages=messages, tools=tools, temperature=0.0, max_tokens=4096 if think else 1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}}, **kw)
    return r.choices[0].message, r.usage.completion_tokens, time.perf_counter() - t0


async def run_one(client, model, item, cond, pool):
    rng = random.Random(f"{item['id']}-{cond}")
    tools, names, kw = build(item, cond, pool, rng)
    messages = [dict(m) for turn in item["question"] for m in turn]
    think = cond == "think"
    msg, ntok, secs = await ask(client, model, messages, tools, think, kw)
    calls, problems = parse_calls(msg)
    retried = False
    if cond == "repair" and calls:
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        errors = list(problems)
        for name, args in calls:
            if name not in schemas:
                errors.append(f"unknown tool '{name}'; available: {', '.join(schemas)}")
            elif args is not None:
                errors += [f"{name}: {e}" for e in validate(args, schemas[name])]
        if errors:
            retried = True
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name,
                                                               "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": "Error: " + "; ".join(errors) + ". Fix the call and try again."})
            msg, ntok2, secs2 = await ask(client, model, messages, tools, False, kw)
            ntok += ntok2
            secs += secs2
            calls, problems = parse_calls(msg)
    s = score(item, [(n, a) for n, a in calls if a is not None], names) if not problems else 0.0
    if problems and item["category"] == "irrelevance":
        s = 0.0
    return {"id": item["id"], "category": item["category"], "cond": cond, "score": s, "n_calls": len(calls),
            "calls": [[n, a] for n, a in calls], "n_tools": len(tools), "tokens": ntok, "secs": round(secs, 3),
            "retried": retried, "content": (msg.content or "")[:2000]}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="local")
    p.add_argument("--conds", default=",".join(CONDITIONS))
    p.add_argument("--live-multiple-n", type=int, default=300)
    p.add_argument("--limit", type=int, default=None, help="cap per category, for smoke tests")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--duty", type=float, default=0.5)
    args = p.parse_args()

    items = []
    for cat in CATEGORIES:
        n = args.live_multiple_n if cat == "live_multiple" else None
        got = [it for it in load(cat, n) if cat == "irrelevance" or it["ground_truth"]]
        items += got[: args.limit] if args.limit else got
    pool = [f for it in items for f in it["function"]]

    out = Path("results") / f"{args.label}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["id"], r["cond"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if r}
    jobs = [(it, c) for c in args.conds.split(",") for it in items if (it["id"], c) not in done]
    total = len(jobs) + len(done)
    print(f"{args.label}: {len(done)} done, {len(jobs)} to go", flush=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="local", timeout=900)

    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(jobs), args.batch):
            chunk = jobs[start:start + args.batch]
            t0 = time.perf_counter()
            res = await asyncio.gather(*(run_one(client, args.model, it, c, pool) for it, c in chunk),
                                       return_exceptions=True)
            busy = time.perf_counter() - t0
            for (it, c), r in zip(chunk, res):
                if isinstance(r, Exception):
                    r = {"id": it["id"], "category": it["category"], "cond": c, "score": 0.0, "error": repr(r)[:500]}
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"{args.label}: {len(done) + start + len(chunk)}/{total} ({busy:.1f}s batch)", flush=True)
            if args.duty < 1:
                await asyncio.sleep(busy * (1 - args.duty) / args.duty)
    print(f"{args.label}: finished", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
