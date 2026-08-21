"""Benchmark: the from-scratch deep neural network. Local GPU/CPU, no API cost.

Trains on ed-donner/items_full (800,000 products) but evaluates on the SAME 200
held-out products as every other model, so the comparison is like for like.

Verified before relying on this: items_lite's test split has zero overlap with
items_full's train split, so training on the full set does not leak the answers
into the evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import load_eval_items, load_split
from harness import run
from pricer.deep_neural_network import DeepNeuralNetworkRunner

# The scheduler is CosineAnnealingLR(T_max=10), so 10 epochs completes exactly
# one cosine cycle and lets the learning rate anneal to zero.
EPOCHS = 10
TRAIN_DATASET = "ed-donner/items_full"
WEIGHTS = Path(__file__).parent / "dnn_weights.pt"

print(f"loading {TRAIN_DATASET} (this needs a few GB of RAM)...", flush=True)
train = load_split("train", TRAIN_DATASET)
val = load_split("validation", TRAIN_DATASET)
print(f"train={len(train):,} val={len(val):,}", flush=True)

runner = DeepNeuralNetworkRunner(train, val)
runner.setup()
print(f"batches per epoch: {len(runner.train_loader):,}", flush=True)

runner.train(epochs=EPOCHS)
runner.save(WEIGHTS)
print(f"saved weights -> {WEIGHTS}", flush=True)

# Evaluation slice comes from items_lite's test split -- identical products to
# the LLM benchmarks.
items = load_eval_items()
run(
    "Deep Neural Network",
    runner.inference,
    items,
    workers=1,
    notes=f"HashingVectorizer(5000) + 289M-param residual MLP, {EPOCHS} epochs on "
    f"{len(train):,} items, trained locally on an RTX 5070 Ti",
)
