"""Benchmark: GPT-4.1-nano, prompt only (no fine-tuning).

Uses the paid OpenAI key. Runs locally only -- this model is never exposed to the
public demo. Token usage is tallied so the real cost is reported, not guessed.
"""

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from openai import OpenAI

from harness import load_eval_items, run

load_dotenv(override=True)
client = OpenAI()

MODEL = "gpt-4.1-nano-2025-04-14"
# https://openai.com/api/pricing -- gpt-4.1-nano
USD_PER_M_IN, USD_PER_M_OUT = 0.10, 0.40

PROMPT = "Estimate the price of this product. Respond with only the price in dollars, no explanation.\n\n{}"

_usage = {"in": 0, "out": 0}
_lock = threading.Lock()


def make_predictor(model: str):
    def predict(item):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(item.summary)}],
            max_tokens=10,
            temperature=0,
        )
        with _lock:
            _usage["in"] += r.usage.prompt_tokens
            _usage["out"] += r.usage.completion_tokens
        return r.choices[0].message.content or "0"

    return predict


def report_cost(label: str, in_rate=USD_PER_M_IN, out_rate=USD_PER_M_OUT):
    cost = _usage["in"] / 1e6 * in_rate + _usage["out"] / 1e6 * out_rate
    print(
        f"{label} token usage: {_usage['in']:,} in / {_usage['out']:,} out "
        f"=> ${cost:.4f}",
        flush=True,
    )
    return cost


if __name__ == "__main__":
    items = load_eval_items()
    run(
        "GPT-4.1-nano (prompted)",
        make_predictor(MODEL),
        items,
        workers=6,
        notes="prompt-only, no training, temperature=0",
    )
    report_cost("GPT-4.1-nano prompted")
