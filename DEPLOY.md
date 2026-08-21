# Deploying the demo

Target: **Vercel**, free Hobby plan. The demo is a static page plus one
zero-dependency Python serverless function.

Why Vercel rather than Hugging Face Spaces: this app is request/response
(~0.4s per estimate), not a persistent process, and the deployed bundle is
tiny because `torch` never ships. Free Spaces sleep after ~48h and take ~30s
to wake, which is a poor property for a portfolio link.

---

## What actually gets deployed

Four things, enforced by [`.vercelignore`](.vercelignore):

```
index.py          WSGI entrypoint - routes /api/predict, serves the two files
index.html        the demo page
showcase.json     pre-computed predictions
api/predict.py    request handling logic (stdlib only)
```

Everything else — `.env`, `pricer/`, `benchmarks/`, `tests/`, the notebooks,
the trained weights — is excluded. The training code is not needed at runtime
and must not sit on a public server.

**Why an `index.py` rather than `api/` functions.** Vercel's current Python
runtime looks for a single application object in one of a few default filenames
(`app.py`, `index.py`, `server.py`, `main.py`, `wsgi.py`, `asgi.py`). The older
"every file under `api/` becomes its own function" convention is no longer
detected, and a deployment without a recognised entrypoint fails the build with:

```
No python entrypoint found in default locations, but found potential
entrypoints: api/predict.py (variable: handler)
```

`index.py` is that entrypoint. It stays dependency-free — plain WSGI, no
framework — and serves static assets from an explicit allowlist so no request
can reach source files or `.env`.

---

## Step 1 — verify the key policy locally

Run this **before every deploy**. It asserts that a visitor cannot spend the
paid OpenAI key, by putting `OPENAI_API_KEY` in the environment and confirming
gated models still refuse to run.

```bash
python tests/test_key_policy.py
```

Expected: `All key-policy checks passed.` Do not deploy if anything fails.

---

## Step 2 — simulate the deployment locally

```bash
python dev_server.py          # http://127.0.0.1:8000
```

Check by hand:

1. The showcase reveals prices with no network calls.
2. "Try your own" returns an estimate on the free model.
3. Selecting a `(need your own key)` model **without** a key is refused.
4. Pasting a Groq key into an OpenAI model is rejected before any request.

---

## Step 3 — install the CLI and authenticate

Vercel's CLI is not installed on this machine.

```bash
npm i -g vercel
vercel login
```

You authenticate yourself; the token stays on your machine.

---

## Step 4 — deploy

From the repository root:

The project name comes from the folder name, which Vercel requires to be
lowercase — this repository's folder is not, so link it explicitly first:

```bash
vercel link --yes --project price-predictor
vercel --yes            # preview deployment
vercel --prod --yes     # promote to production
```

Without the `link` step the deploy fails with *"Project names ... must be
lowercase"*. No `vercel.json` is required.

---

## Step 5 — turn off Deployment Protection

New Vercel projects enable **Vercel Authentication** by default, which puts a
login wall in front of every deployment — including production. The site returns
`401 Protected deployment` to anyone not signed in to the Vercel account, which
defeats the point of a public demo.

**Project → Settings → Deployment Protection → Vercel Authentication →
Disabled**, then save.

---

## Step 6 — add the secret

In the Vercel dashboard: **Project → Settings → Environment Variables**

| Name | Value | Environments |
|------|-------|--------------|
| `GROQ_API_KEY` | your Groq key | Production, Preview |

**Add only this one.** Do not add `OPENAI_API_KEY` — the demo has no code path
that would use it, and adding it would put a paid credential on a public server
for no reason.

Redeploy after adding (`vercel --prod`) so the function picks it up.

---

## Step 7 — verify the live deployment

Replace `<url>` with the deployed domain.

**The free model works:**

```bash
curl -s -X POST https://<url>/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"description":"Sony WH-1000XM4 wireless headphones, 30-hour battery"}'
```

Expect `{"price": ..., "byok": false, ...}`.

**A gated model is refused — this is the important one:**

```bash
curl -s -X POST https://<url>/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"description":"a laptop","model":"gpt-4.1-nano"}'
```

Expect HTTP 400 and `{"error":"key_required", ...}`. If this ever returns a
price, the paid key has leaked into the deployment — remove
`OPENAI_API_KEY` from the Vercel environment immediately.

**No secrets in the bundle:**

```bash
curl -s https://<url>/.env          # expect 404
curl -s https://<url>/api/predict | grep -i "gsk_\|sk-"   # expect no matches
```

---

## Known platform notes

- **Groq is behind Cloudflare.** Raw `urllib` calls with the default
  `Python-urllib/x.y` User-Agent get `403 error code: 1010`, which looks exactly
  like an auth failure. `api/predict.py` sends an explicit `User-Agent`; do not
  remove it.
- **Groq free tier is 30 requests/minute for the whole organisation.** The
  showcase is pre-computed specifically so the demo never depends on that quota.
  Live requests that hit it return a friendly message and retry once.
- **`gpt-oss` bills hidden reasoning tokens against `max_tokens`.** Below ~200
  the model can return `''` or a bare `'$'`. The endpoint rejects any reply
  without a digit rather than rendering `$0`.
- **Hobby plan prohibits commercial use.** Fine for a portfolio demo.

---

## Regenerating the showcase

If the models or dataset change:

```bash
python benchmarks/run_dnn.py            # trains on 800k items, ~80 min on an RTX 5070 Ti
python benchmarks/run_openai_prompted.py
python benchmarks/run_groq.py
python benchmarks/generate_showcase.py  # writes showcase.json
```
