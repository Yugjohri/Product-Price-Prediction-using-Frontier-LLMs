"""Fine-tune GPT-4.1-nano on price estimation, then benchmark it.

Deliberately small: fine-tuning cost scales with tokens x epochs, and the point is
to measure whether fine-tuning helps at all, not to win a leaderboard.

The resulting model is used for LOCAL BENCHMARKING ONLY. It is never wired into
the public demo -- that runs on the free Groq model.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from openai import OpenAI

from harness import load_eval_items, load_split, run
from run_openai_prompted import make_predictor, report_cost

load_dotenv(override=True)
client = OpenAI()

BASE_MODEL = "gpt-4.1-nano-2025-04-14"
N_TRAIN, N_VAL = 500, 100
JSONL_DIR = Path(__file__).resolve().parent.parent / "jsonl"
MODEL_RECORD = Path(__file__).parent / "finetuned_model.txt"

PROMPT = "Estimate the price of this product. Respond with only the price in dollars, no explanation.\n\n{}"


def messages_for(item):
    return [
        {"role": "user", "content": PROMPT.format(item.summary)},
        {"role": "assistant", "content": f"${item.price:.2f}"},
    ]


def write_jsonl(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps({"messages": messages_for(item)}) + "\n")
    print(f"wrote {len(items)} examples -> {path}", flush=True)


def main():
    train = load_split("train")
    val = load_split("validation")

    train_path = JSONL_DIR / "ft_train.jsonl"
    val_path = JSONL_DIR / "ft_val.jsonl"
    write_jsonl(train[:N_TRAIN], train_path)
    write_jsonl(val[:N_VAL], val_path)

    with train_path.open("rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    with val_path.open("rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    print(f"uploaded train={train_file.id} val={val_file.id}", flush=True)

    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=BASE_MODEL,
        seed=42,
        suffix="pricer",
    )
    print(f"created fine-tune job {job.id}", flush=True)

    # Poll until it finishes.
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        print(f"  status={job.status} ...", flush=True)
        time.sleep(30)

    print(f"job finished: {job.status}", flush=True)
    if job.status != "succeeded":
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=10).data
        for e in events:
            print("  ", e.message, flush=True)
        sys.exit(1)

    model_name = job.fine_tuned_model
    trained_tokens = job.trained_tokens or 0
    MODEL_RECORD.write_text(model_name, encoding="utf-8")
    print(f"fine-tuned model: {model_name}", flush=True)
    # gpt-4.1-nano fine-tuning training rate
    print(f"trained_tokens={trained_tokens:,} => training cost ~${trained_tokens / 1e6 * 1.50:.4f}", flush=True)

    items = load_eval_items()
    run(
        "GPT-4.1-nano (fine-tuned)",
        make_predictor(model_name),
        items,
        workers=6,
        notes=f"fine-tuned on {N_TRAIN} examples, seed=42, auto hyperparameters",
    )
    # Fine-tuned nano inference is billed at 2x the base rate.
    report_cost("GPT-4.1-nano fine-tuned", in_rate=0.20, out_rate=0.80)


if __name__ == "__main__":
    main()
