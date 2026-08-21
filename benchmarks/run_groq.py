"""Benchmark: free open-weights model via Groq. This is the model the live demo serves."""

import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from groq import Groq

from harness import load_eval_items, run

load_dotenv(override=True)
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"
# NOTE: gpt-oss is a reasoning model. Without reasoning_effort="low" it spends the
# whole token budget on hidden reasoning and returns an EMPTY string.
REASONING_EFFORT = "low"

# Groq free tier allows 30 requests/minute for this model. Pace below that.
RPM_LIMIT = 26
_lock = threading.Lock()
_next_slot = [0.0]


def _wait_turn():
    """Global pacer: never issue requests faster than RPM_LIMIT."""
    interval = 60.0 / RPM_LIMIT
    with _lock:
        now = time.monotonic()
        start = max(now, _next_slot[0])
        _next_slot[0] = start + interval
    delay = start - time.monotonic()
    if delay > 0:
        time.sleep(delay)


PROMPT = "Estimate the price of this product. Respond with only the price in dollars, no explanation.\n\n{}"


# Reasoning tokens share this budget with the visible answer. Too small and the
# model burns it all reasoning, returning '' or a bare '$' -- which parses to $0
# and silently poisons the metrics. 200 is comfortably above what "low" needs.
MAX_TOKENS = 200


def predict(item, retries=8):
    for attempt in range(retries):
        _wait_turn()
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT.format(item.summary)}],
                max_tokens=MAX_TOKENS,
                reasoning_effort=REASONING_EFFORT,
                temperature=0,
            )
            content = r.choices[0].message.content or ""
            # A usable answer must contain a digit; a truncated '$' does not.
            if re.search(r"\d", content):
                return content
            if attempt == retries - 1:
                raise ValueError(f"no numeric answer after {retries} tries: {content!r}")
        except Exception as e:
            if attempt == retries - 1:
                raise
            # Honour the server's own "try again in Xs" hint when present.
            hint = re.search(r"try again in ([\d.]+)s", str(e))
            time.sleep(float(hint.group(1)) + 0.5 if hint else 2**attempt)
    return "0"


if __name__ == "__main__":
    items = load_eval_items()
    run(
        f"Groq {MODEL}",
        predict,
        items,
        workers=3,
        notes=f"prompt-only, no training, reasoning_effort={REASONING_EFFORT}, "
        f"free tier paced to {RPM_LIMIT} req/min",
    )
