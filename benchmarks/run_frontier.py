"""Benchmark any model from the demo's registry, through the demo's own code path.

Deliberately calls api.predict.call_model rather than reimplementing the request,
so a benchmark result is evidence about the deployed endpoint, not about a
parallel implementation that might drift from it.

    python benchmarks/run_frontier.py gpt-5-nano
"""

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT / ".env", override=True)

from api import predict as api
from harness import load_eval_items, run

# Rough public rates, $ per 1M tokens, for reporting measured cost.
RATES = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.1": (1.25, 10.00),
}

_usage = {"in": 0, "out": 0}
_lock = threading.Lock()


def make_predictor(model_id: str):
    spec = api.MODELS[model_id]
    provider = api.PROVIDERS[spec["provider"]]
    key = os.environ["GROQ_API_KEY"] if spec["provider"] == "groq" else os.environ["OPENAI_API_KEY"]

    def predict(item):
        raw, _ = api.call_model(spec, key, item.summary)
        price = api.parse_price(raw)
        if price is None:
            raise ValueError(f"no price in reply: {raw!r}")
        return price

    return predict, spec


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt-5-nano"
    if model_id not in api.MODELS:
        sys.exit(f"unknown model {model_id}; choose from {list(api.MODELS)}")

    predictor, spec = make_predictor(model_id)
    items = load_eval_items()
    label = spec["label"].replace("(need your own key) ", "").split(" - ")[0]

    run(
        label,
        predictor,
        items,
        workers=4,
        notes=f"prompt-only via the deployed endpoint's code path, "
        f"reasoning_effort={spec['reasoning_effort']}",
    )
