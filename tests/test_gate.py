import random

from toolcall.gate import auroc, cv, evaluate


def test_auroc_separates():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0


def test_auroc_ties_count_half():
    assert auroc([0.0], [0.0]) == 0.5
    assert auroc([0.0, 0.0, 1.0], [0.0, 0.0]) == 2 / 3
    # swapping the classes flips the score, ties included
    pos, neg = [0.0] * 5 + [1.0], [0.0] * 3 + [0.5]
    assert auroc(pos, neg) + auroc(neg, pos) == 1.0


def test_auroc_matches_pairwise_count():
    rng = random.Random(0)
    pos = [rng.choice([0.0, 0.1, 0.2, 0.5]) for _ in range(40)]
    neg = [rng.choice([0.0, 0.1, 0.3]) for _ in range(30)]
    pairs = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    assert abs(auroc(pos, neg) - pairs / (len(pos) * len(neg))) < 1e-12


def item(irr=False, s0=1, s1=1, conf=0.0, p_call=1.0):
    return {"irr": irr, "s0": s0, "s1": s1, "t0": 10, "t1": 100, "p_call": p_call, "conf": conf, "called": True}


def test_evaluate_routes_each_request():
    items = [item(irr=True, p_call=0.0), item(s0=1, conf=-0.1), item(s0=0, s1=1, conf=-2.0)]
    acc, cost, thinks = evaluate(items, ta=0.05, tc=-1.0)
    assert acc == 1.0
    assert thinks == 1 / 3
    assert cost == (10 * 3 + 100) / 3


def test_cv_is_deterministic():
    rng = random.Random(1)
    items = [item(s0=rng.random() < 0.7, s1=rng.random() < 0.8, conf=-rng.random()) for _ in range(50)]
    assert cv(items) == cv(items)
