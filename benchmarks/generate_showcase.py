"""Pre-compute predictions for the demo's showcase panel.

The showcase must work instantly and never fail, so nothing in it is computed at
request time. Every model runs here, once, on a fixed set of test products, and
the answers ship with the site as a static JSON file.

Run AFTER run_dnn.py, since it loads the trained weights.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import RESULTS_PATH, load_eval_items, load_split, post_process
from pricer.deep_neural_network import DeepNeuralNetworkRunner
from run_groq import predict as groq_predict
from run_openai_prompted import make_predictor as openai_predictor

SHOWCASE_SIZE = 60
OUT = Path(__file__).resolve().parent.parent / "showcase.json"
WEIGHTS = Path(__file__).parent / "dnn_weights.pt"


def build_dnn():
    """Rebuild the vectorizer + normalisation stats, then load trained weights."""
    runner = DeepNeuralNetworkRunner(load_split("train"), load_split("validation"))
    runner.setup()
    runner.load(WEIGHTS)
    return runner


def main():
    items = load_eval_items(SHOWCASE_SIZE)
    print(f"showcase items: {len(items)}", flush=True)

    print("running DNN...", flush=True)
    dnn = build_dnn()
    dnn_guesses = [round(dnn.inference(i), 2) for i in items]

    print("running GPT-4.1-nano...", flush=True)
    gpt = openai_predictor("gpt-4.1-nano-2025-04-14")
    gpt_guesses = [post_process(gpt(i)) for i in items]

    print("running Groq gpt-oss-20b (rate-paced, ~3 min)...", flush=True)
    groq_guesses = [post_process(groq_predict(i)) for i in items]

    payload = {
        "eval_size": len(items),
        "aggregate": json.loads(RESULTS_PATH.read_text(encoding="utf-8")),
        "items": [
            {
                "title": it.title[:110],
                "summary": (it.summary or "").strip(),
                "category": it.category,
                "price": round(it.price, 2),
                "guesses": {"groq": g, "gpt41nano": p, "dnn": d},
            }
            for it, g, p, d in zip(items, groq_guesses, gpt_guesses, dnn_guesses)
        ],
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
