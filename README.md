# What does this cost?

**Estimating product prices from text — comparing a prompted LLM against a neural
network trained from scratch.**

[![Try it live](https://img.shields.io/badge/Try_it_live-price--predictor-black?logo=vercel&logoColor=white)](https://price-predictor-yug20.vercel.app)

Predicting a price is *regression* — mapping text onto a continuous number.
Language models are next-token predictors and were never built for it. So: can
they actually do it, and does a purpose-built model beat them?

Every number below was measured by the code in this repository, on one machine,
on the same 200 held-out products.

---

## Results

| Approach | Avg error | Within 20% | RMSLE | Cost per 200 estimates |
|---|---:|---:|---:|---:|
| **Deep neural network** — trained from scratch | **$46.77** | **67.0%** | 0.533 | $0 (local, ~80 min GPU) |
| **GPT-4.1 nano** — prompt only | $63.95 | 60.0% | 0.628 | $0.0024 |
| **GPT-5 nano** — prompt only | $77.84 | 60.0% | 0.697 | ~$0.012 |
| **GPT-OSS 20B** — prompt only, open weights | $83.95 | 53.0% | 0.788 | $0 (free tier) |
| ~~GPT-4.1 nano, fine-tuned~~ | — | — | — | *no longer possible — see below* |

**A 289M-parameter network trained from scratch beats every frontier model
tested — by 27% against the best of them.** It is also roughly 25× faster at
inference (0.018s vs 0.127s per estimate) and costs nothing to run.

That is the answer to the question this project asks. A language model can
estimate prices creditably with no training at all, which is remarkable. But on
a narrow, well-specified regression task with 800,000 labelled examples
available, a purpose-built model still wins clearly.

### Three things the numbers contradict

**1. Newer is not better here.** GPT-5 nano is a generation ahead of GPT-4.1 nano
and lands 22% *worse* on average error. Identical hit rate (60%), but far bigger
outliers. For a task this narrow, reasoning capability isn't the bottleneck —
price priors are.

**2. Cheaper per token is not cheaper per job.** GPT-5 nano has *half* the input
price of GPT-4.1 nano, and costs roughly **12× more per estimate**:

| | Input rate | Tokens per estimate | Actual cost per call |
|---|---:|---:|---:|
| GPT-4.1 nano | $0.10/M | 41 in, **2 out** | **$0.0000049** |
| GPT-5 nano | $0.05/M | 40 in, **141 out** | $0.0000584 |

Reasoning models bill their hidden reasoning as output tokens. Per-token
headline rates are close to meaningless without knowing how many tokens a model
actually spends.

**3. Training data mattered far more than architecture or schedule.** The same
network, unchanged, on three configurations:

| Training set | Hardware | Result |
|---|---|---:|
| 20,000 items, 5 epochs | CPU | $70.45 |
| 20,000 items, 10 epochs | CPU | $66.89 |
| **800,000 items, 10 epochs** | **GPU** | **$46.77** |

The 20k runs were badly overfitting — final train loss 0.1414 against validation
0.5304, a 3.75× gap. A 289M-parameter model has far more capacity than 20,000
examples can constrain, so it memorised them. With 40× the data the gap inverts
(train 0.53, validation 0.43) and the error drops by a third. The lesson is that
the overfitting signature was visible in the logs the whole time, and worth
reading before trusting any headline number.

---

## The fine-tuning story

This project originally compared a **fourth** approach: fine-tuning GPT-4.1 nano
on the price data. That comparison can no longer be reproduced by anyone who
hasn't already run one.

On **7 May 2026**, OpenAI began winding down its self-serve fine-tuning platform.
The cutoff was retroactive — only organisations that had *already* run a
fine-tuning job kept access. Running `benchmarks/run_finetune.py` today returns:

```
403 - OpenAI is winding down the fine-tuning platform and your organization
is no longer able to create new fine-tuning training jobs.
code: training_not_available
```

The phased timeline, from
[OpenAI's deprecation notice](https://developers.openai.com/api/docs/deprecations):

| Date | Effect |
|---|---|
| 7 May 2026 | Organisations that never fine-tuned lose the ability to start |
| 2 July 2026 | Extends to those with no fine-tuned inference in 60 days |
| 6 Jan 2027 | Even active customers can no longer create new jobs |

**The script is kept in this repository on purpose.** A capability that a vendor
can withdraw is not a capability you own, and that is worth demonstrating rather
than quietly deleting. The failure is reproducible; the dependency risk is the
finding.

---

## Live demo

A two-part page: a **pre-computed showcase** that steps through real test
products and reveals each model's guess against the true price, plus a **live
estimator** that really calls a model.

The showcase is pre-computed deliberately. Groq's free tier allows 30
requests/minute *for the whole account*, so a demo that called the API on every
interaction would break exactly when it got shared. Splitting it means the part
that has to impress always works, and only custom input touches the network.

**Key policy.** The free model runs on the host's Groq key. Every other model
requires the visitor to supply their own — and the gated code path never reads
`os.environ` at all, so there is no branch in which a premium model can fall back
to the host's credentials. Asserted, not assumed:

```bash
python tests/test_key_policy.py
```

That test puts `OPENAI_API_KEY` into the environment and proves a gated request
*still* refuses to run and never reaches upstream.

---

## Two bugs worth knowing about

**Reasoning tokens share the answer's budget — on every provider tested.**
`gpt-oss` and the GPT-5 family both draw hidden reasoning from `max_tokens`. Set
it too low and you get an empty string or a bare `$`, with no exception raised:

```
gpt-5-nano, max_completion_tokens=600            -> content=''   (600 tokens, all reasoning)
gpt-5-nano, + reasoning_effort="low"             -> '$249.99'
```

A single truncated `'$'` parsed as `$0` in a 200-item run pushed RMSLE from 0.63
to 1.50 while average error still looked plausible. The endpoint now rejects any
reply without a digit rather than rendering `$0`.

**Groq sits behind Cloudflare.** Raw `urllib` requests using the default
`Python-urllib/x.y` agent get `403 error code: 1010` — indistinguishable from an
auth failure, and it sends you debugging a perfectly good key. An explicit
`User-Agent` fixes it.

---

## How it works

```
pricer/parser.py      curation: price bands, part-number stripping,
                      weight normalisation across five unit systems
pricer/loaders.py     multiprocess loading over the raw dataset
pricer/items.py       the Item model and Hub round-tripping
pricer/evaluator.py   scoring: error bands, 95% CI convergence charts
pricer/deep_neural_network.py
                      289M-param residual MLP over hashed text features
pricer/batch.py       batch-API preprocessing (50% cheaper than serial calls)

api/predict.py        the deployed endpoint — stdlib only, zero dependencies
benchmarks/           every number in this README, reproducible
tests/test_key_policy.py
                      proves visitors cannot spend the host's paid key
```

The network trains on `log(price)` rather than price. Prices are heavily
right-skewed — many $20 items, few $900 ones — so training on the raw target
lets the rare expensive items dominate the gradient.

---

## Running it

```bash
python -m venv .venv                       # Python 3.12 (3.14 has no torch wheels)
.venv/Scripts/pip install -r requirements-dev.txt
cp .env.example .env                       # add GROQ_API_KEY (free)

python dev_server.py                       # demo at http://127.0.0.1:8000
jupyter lab PricePredict.ipynb             # pipeline walkthrough, free tier
```

For GPU training, install a CUDA build of torch rather than the default wheel —
`pip install torch` gives you the CPU-only build on Windows, which trains this
model roughly 18× slower:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Reproduce the benchmarks:

```bash
python benchmarks/run_groq.py              # free
python benchmarks/run_openai_prompted.py   # ~$0.002
python benchmarks/run_frontier.py gpt-5-nano
python benchmarks/run_dnn.py               # free, ~80 min on an RTX 5070 Ti
```

Deployment steps and verification: [DEPLOY.md](DEPLOY.md).

---

## Data

Curated Amazon products with cleaned summaries and ground-truth prices, derived
from [McAuley-Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
Both splits are public — no Hugging Face token required.

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| [`items_full`](https://huggingface.co/datasets/ed-donner/items_full) — network training | 800,000 | 10,000 | 10,000 |
| [`items_lite`](https://huggingface.co/datasets/ed-donner/items_lite) — evaluation slice | 20,000 | 1,000 | 1,000 |

**Evaluation uses the first 200 items of `items_lite`'s test split for every
model** — identical products, no shuffling, so the comparison is like for like.

The network trains on `items_full` but is scored on that same slice. Verified
before relying on it: `items_lite`'s test split has **zero** overlap with
`items_full`'s train split (all 1,000 lite test items live in full's *test*
split), so training on the larger set does not leak the evaluation answers.
