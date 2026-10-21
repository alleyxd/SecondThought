"""BFCL data loading, schema conversion and an AST-style answer checker.

The checker follows BFCL's rules: the right function, every required argument present, every provided argument
within the list of accepted values (an empty string in that list means the argument may be omitted), and no
arguments the ground truth doesn't know about. Strings are compared after lowercasing and dropping whitespace and
punctuation, numbers numerically.
"""
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
CATEGORIES = {
    "simple": "BFCL_v3_simple",
    "multiple": "BFCL_v3_multiple",
    "live_simple": "BFCL_v3_live_simple",
    "live_multiple": "BFCL_v3_live_multiple",
    "irrelevance": "BFCL_v3_irrelevance",
}
TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": None}


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load(category: str, n: int | None = None, seed: int = 0) -> list[dict]:
    name = CATEGORIES[category]
    items = _lines(DATA / f"{name}.json")
    answers = {}
    ans_path = DATA / f"ans_{name}.json"
    if ans_path.exists():
        answers = {a["id"]: a["ground_truth"] for a in _lines(ans_path)}
    for it in items:
        it["category"] = category
        it["ground_truth"] = answers.get(it["id"])
    if n is not None and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def convert_schema(s):
    """BFCL uses Python-ish type names; turn them into JSON Schema."""
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            if k == "type" and isinstance(v, str):
                t = TYPE_MAP.get(v, v)
                if t is None:
                    continue
                out[k] = t
            elif k in ("properties",) and isinstance(v, dict):
                out[k] = {pk: convert_schema(pv) for pk, pv in v.items()}
            else:
                out[k] = convert_schema(v)
        return out
    if isinstance(s, list):
        return [convert_schema(x) for x in s]
    return s


def to_tool(fn: dict, describe: bool = True, describe_params: bool = True, name: str | None = None) -> dict:
    params = convert_schema(fn.get("parameters", {"type": "object", "properties": {}}))
    if not describe_params:
        params = json.loads(json.dumps(params))
        for p in params.get("properties", {}).values():
            p.pop("description", None)
    return {"type": "function", "function": {
        "name": name or safe_name(fn["name"]),
        "description": fn.get("description", "") if describe else "",
        "parameters": params,
    }}


# ---------------------------------------------------------------- checking

def _norm_str(s: str) -> str:
    return re.sub(r"[\s,./\-_*^'\"]", "", s.lower())


def _eq(v, p) -> bool:
    if isinstance(p, bool) or isinstance(v, bool):
        return v == p
    if isinstance(p, (int, float)):
        try:
            return float(v) == float(p)
        except (TypeError, ValueError):
            return False
    if isinstance(p, str):
        return isinstance(v, str) and _norm_str(v) == _norm_str(p)
    if isinstance(p, list):
        return isinstance(v, list) and len(v) == len(p) and all(_eq(a, b) for a, b in zip(v, p))
    if isinstance(p, dict):
        if not isinstance(v, dict):
            return False
        for k, poss in p.items():
            if k in v:
                if not _match(v[k], poss):
                    return False
            elif "" not in poss and poss != [""]:
                return False
        return all(k in p for k in v)
    return v == p


def _match(v, possible) -> bool:
    if not isinstance(possible, list):
        possible = [possible]
    return any(_eq(v, p) for p in possible if p != "")


def check_call(name: str, args: dict, truth: dict, names: dict) -> bool:
    """truth: {original_function_name: {param: [accepted values]}}; names maps sent name -> original name."""
    (gt_name, gt_args), = truth.items()
    if names.get(name, name) != gt_name:
        return False
    if not isinstance(args, dict):
        return False
    for k, v in args.items():
        if k not in gt_args or not _match(v, gt_args[k]):
            return False
    for k, poss in gt_args.items():
        if k not in args and "" not in poss:
            return False
    return True


def score(item: dict, calls: list[tuple[str, dict]], names: dict) -> float:
    if item["category"] == "irrelevance":
        return float(len(calls) == 0)
    truths = item["ground_truth"] or []
    if len(calls) != len(truths):
        return 0.0
    # BFCL simple/multiple/live have one expected call; allow any order for safety
    remaining = list(truths)
    for name, args in calls:
        hit = next((t for t in remaining if check_call(name, args, t, names)), None)
        if hit is None:
            return 0.0
        remaining.remove(hit)
    return 1.0


# ---------------------------------------------------------------- schema validation for the repair condition

_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(args, schema: dict) -> list[str]:
    errors = []
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            errors.append(f"missing required argument '{req}'")
    for k, v in args.items():
        if k not in props:
            errors.append(f"unknown argument '{k}'; allowed: {', '.join(props) or 'none'}")
            continue
        t = props[k].get("type")
        py = _PY.get(t)
        if py and not (isinstance(v, py) and not (t in ("integer", "number") and isinstance(v, bool))):
            errors.append(f"argument '{k}' should be of type {t}, got {type(v).__name__}")
        if "enum" in props[k] and v not in props[k]["enum"]:
            errors.append(f"argument '{k}' must be one of {props[k]['enum']}")
    return errors
