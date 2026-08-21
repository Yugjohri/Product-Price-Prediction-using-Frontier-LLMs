"""Price estimation endpoint for the public demo.

Deliberately zero-dependency (stdlib only) so the deployed bundle stays tiny and
there is no SDK version drift to manage in production.

KEY POLICY -- the single most important thing in this file:

    Free-tier models run on the HOST's Groq key.
    Every other model REQUIRES the visitor to supply their own key.

The gated branch never reads os.environ at all, so there is no code path in
which a premium model can silently fall back to the host's credentials. This is
enforced *before* any key is read, not by checking a spend limit afterwards.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
# The "(need your own key)" warning leads the label: a narrow <select> truncates
# the END of an option, so a trailing marker is exactly the part that vanishes.

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "host_key_env": "GROQ_API_KEY",
        "key_prefix": "gsk_",
        "console": "https://console.groq.com/keys",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "host_key_env": None,  # never runs on the host key
        "key_prefix": "sk-",
        "console": "https://platform.openai.com/api-keys",
    },
}

# "token_param" exists because the families disagree: GPT-5 rejects max_tokens
# outright with HTTP 400 and demands max_completion_tokens.
#
# "reasoning_effort" is not optional tuning on reasoning models -- it is load
# bearing. Without it, gpt-oss AND the GPT-5 family spend the entire token
# budget on hidden reasoning and return an EMPTY string with finish=length.
# Measured: gpt-5-nano burned 600 tokens and returned '' until effort was set.

MODELS = {
    "gpt-oss-20b": {
        "label": "GPT-OSS 20B - free, runs on my key",
        "provider": "groq",
        "api_model": "openai/gpt-oss-20b",
        "tier": "free",
        "reasoning_effort": "low",
        "token_param": "max_tokens",
        "temperature": 0,
    },
    "gpt-oss-120b": {
        "label": "(need your own key) GPT-OSS 120B",
        "provider": "groq",
        "api_model": "openai/gpt-oss-120b",
        "tier": "byok",
        "reasoning_effort": "low",
        "token_param": "max_tokens",
        "temperature": 0,
    },
    "gpt-4.1-nano": {
        "label": "(need your own key) GPT-4.1 nano - cheapest per estimate",
        "provider": "openai",
        "api_model": "gpt-4.1-nano-2025-04-14",
        "tier": "byok",
        "reasoning_effort": None,
        "token_param": "max_tokens",
        "temperature": 0,
    },
    "gpt-5-nano": {
        "label": "(need your own key) GPT-5 nano",
        "provider": "openai",
        "api_model": "gpt-5-nano",
        "tier": "byok",
        "reasoning_effort": "low",
        "token_param": "max_completion_tokens",
        "temperature": None,  # GPT-5 only accepts the default
    },
    "gpt-5-mini": {
        "label": "(need your own key) GPT-5 mini",
        "provider": "openai",
        "api_model": "gpt-5-mini",
        "tier": "byok",
        "reasoning_effort": "low",
        "token_param": "max_completion_tokens",
        "temperature": None,
    },
    "gpt-5.1": {
        "label": "(need your own key) GPT-5.1 - frontier",
        "provider": "openai",
        "api_model": "gpt-5.1",
        "tier": "byok",
        "reasoning_effort": "low",
        "token_param": "max_completion_tokens",
        "temperature": None,
    },
}

DEFAULT_MODEL = "gpt-oss-20b"

PROMPT = (
    "Estimate the price of this product. Respond with only the price in dollars, "
    "no explanation.\n\n{}"
)

BUSY_MESSAGE = (
    "Demo's a bit busy right now - free tier limits. "
    "Give it a few seconds and try again. - yug"
)

# Reasoning models bill hidden reasoning against this budget, so it must be
# generous. Measured: gpt-5-nano spends ~140 tokens reasoning at effort=low.
MAX_TOKENS = 600
MAX_DESCRIPTION_CHARS = 4000

# Must be a real-looking agent string; see the Cloudflare note in call_model().
USER_AGENT = "price-predictor-demo/1.0"

# Matches anything key-shaped so it can never be echoed back to a caller.
_KEY_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|gsk_[A-Za-z0-9_\-]{8,})")


def scrub(text: str) -> str:
    """Provider SDKs and APIs echo credentials inside error bodies. Never relay them."""
    return _KEY_PATTERN.sub("[redacted]", str(text))


def call_model(spec: dict, api_key: str, description: str) -> tuple[str, float]:
    """One chat completion. Returns (raw_text, elapsed_seconds)."""
    provider = PROVIDERS[spec["provider"]]
    body = {
        "model": spec["api_model"],
        "messages": [{"role": "user", "content": PROMPT.format(description)}],
        spec["token_param"]: MAX_TOKENS,
    }
    if spec["reasoning_effort"]:
        body["reasoning_effort"] = spec["reasoning_effort"]
    if spec["temperature"] is not None:
        body["temperature"] = spec["temperature"]

    req = urllib.request.Request(
        provider["url"],
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Groq sits behind Cloudflare, which rejects urllib's default
            # "Python-urllib/x.y" agent with 403 "error code: 1010". That looks
            # exactly like an auth failure and is deeply misleading to debug.
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"] or "", time.time() - t0


def parse_price(text: str):
    """Pull the first number out of the model's reply, or None if there isn't one."""
    cleaned = text.replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def resolve_key(spec: dict, byok: str):
    """Decide which credential to use. Returns (key, error_dict_or_None).

    A 'byok' model NEVER touches os.environ -- the host key is unreachable from
    that branch by construction.
    """
    provider = PROVIDERS[spec["provider"]]

    if spec["tier"] == "byok":
        if not byok:
            # Use the bare model name: the label's gating prefix and any
            # " - descriptor" suffix both read badly mid-sentence.
            name = spec["label"].replace("(need your own key) ", "").split(" - ")[0]
            return None, {
                "error": "key_required",
                "message": (
                    f"{name} needs your own API key. "
                    f"Paste one below, or switch to the free model."
                ),
                "console": provider["console"],
            }
        if not byok.startswith(provider["key_prefix"]):
            return None, {
                "error": "key_wrong_provider",
                "message": (
                    f"That doesn't look like a {spec['provider']} key "
                    f"(expected it to start with '{provider['key_prefix']}')."
                ),
                "console": provider["console"],
            }
        return byok, None

    # Free tier: host key. A visitor-supplied key is ignored here entirely.
    host_key = os.environ.get(provider["host_key_env"] or "")
    if not host_key:
        return None, {
            "error": "not_configured",
            "message": "The demo isn't configured with a key right now.",
        }
    return host_key, None


def handle(payload: dict) -> tuple[int, dict]:
    description = (payload.get("description") or "").strip()
    model_id = payload.get("model") or DEFAULT_MODEL
    byok = (payload.get("api_key") or "").strip()

    if not description:
        return 400, {"error": "empty", "message": "Describe a product first."}
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS]
    if model_id not in MODELS:
        return 400, {"error": "unknown_model", "message": "Unknown model."}

    spec = MODELS[model_id]
    api_key, err = resolve_key(spec, byok)
    if err:
        return 400, err

    try:
        raw, elapsed = call_model(spec, api_key, description)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        if e.code == 429:
            # Only the free tier shares the host's quota; a BYOK 429 is theirs.
            msg = BUSY_MESSAGE if spec["tier"] == "free" else scrub(detail) or "Rate limited."
            return 429, {"error": "rate_limited", "message": msg, "retry": True}
        if e.code == 401:
            return 401, {
                "error": "bad_key",
                "message": "That key was rejected by the provider.",
            }
        if e.code == 403:
            # Not necessarily auth: Cloudflare returns 403/1010 for a blocked
            # User-Agent. Report it as upstream rather than blaming the key.
            return 502, {
                "error": "upstream",
                "message": scrub(detail) or "The provider refused the request.",
            }
        return 502, {"error": "upstream", "message": scrub(detail) or "Upstream error."}
    except Exception as e:
        return 502, {"error": "upstream", "message": scrub(e)}

    price = parse_price(raw)
    if price is None:
        # Truncated/empty reply -- surface it honestly rather than showing "$0".
        return 502, {
            "error": "no_price",
            "message": "The model didn't return a usable price. Try again.",
        }

    return 200, {
        "price": round(price, 2),
        "model": model_id,
        "model_label": spec["label"],
        "byok": spec["tier"] == "byok",
        "seconds": round(elapsed, 2),
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, obj: dict):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        """Expose the model registry so the UI never hardcodes the list."""
        self._send(
            200,
            {
                "models": [
                    {"id": k, "label": v["label"], "tier": v["tier"], "provider": v["provider"]}
                    for k, v in MODELS.items()
                ],
                "default": DEFAULT_MODEL,
            },
        )

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad_json", "message": "Malformed request."})
        status, body = handle(payload)
        self._send(status, body)

    def log_message(self, fmt, *args):
        """Silence default request logging -- request bodies carry BYOK keys."""
        return
