"""Shared benchmark harness.

Every model is scored on the SAME test items with the SAME metrics, so the
numbers in the README are directly comparable. Results are appended to
benchmarks/results.json.
"""

import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make `pricer` importable no matter where this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from pricer.items import Item

DATASET = "ed-donner/items_lite"
EVAL_SIZE = 200
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_split(split: str, dataset: str = DATASET) -> list[Item]:
    return [Item.model_validate(r) for r in load_dataset(dataset)[split]]


def load_eval_items(size: int = EVAL_SIZE) -> list[Item]:
    """The fixed evaluation slice — first `size` test items, no shuffling."""
    return load_split("test")[:size]


def post_process(value) -> float:
    """Pull a number out of whatever the model said. Mirrors pricer.evaluator."""
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else 0.0
    return float(value)


def score(guesses: list[float], truths: list[float]) -> dict:
    """Average absolute error, RMSLE, and hit rate (within $40 or 20%)."""
    n = len(truths)
    errors = [abs(g - t) for g, t in zip(guesses, truths)]
    sles = [(math.log(max(g, 0) + 1) - math.log(t + 1)) ** 2 for g, t in zip(guesses, truths)]
    hits = sum(1 for e, t in zip(errors, truths) if e < 40 or e / t < 0.2)
    return {
        "n": n,
        "avg_error": round(sum(errors) / n, 2),
        "rmsle": round(math.sqrt(sum(sles) / n), 4),
        "hit_rate_pct": round(100 * hits / n, 1),
    }


def run(name: str, predictor, items: list[Item], workers: int = 1, notes: str = "") -> dict:
    """Run a predictor over the eval items and record the result."""
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()

    def one(item):
        return post_process(predictor(item))

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            guesses = list(ex.map(one, items))
    else:
        guesses = [one(i) for i in items]

    elapsed = time.time() - t0
    truths = [i.price for i in items]
    result = score(guesses, truths)
    result.update(
        {
            "model": name,
            "notes": notes,
            "total_seconds": round(elapsed, 1),
            "seconds_per_item": round(elapsed / len(items), 3),
        }
    )
    print(
        f"avg_error=${result['avg_error']}  rmsle={result['rmsle']}  "
        f"hit_rate={result['hit_rate_pct']}%  ({elapsed:.1f}s total)",
        flush=True,
    )
    save(result)
    return result


def save(result: dict) -> None:
    """Upsert by model name so re-running a benchmark replaces its old row."""
    existing = []
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    existing = [r for r in existing if r["model"] != result["model"]]
    existing.append(result)
    existing.sort(key=lambda r: r["avg_error"])
    RESULTS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
