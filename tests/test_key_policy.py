"""Proves the demo cannot spend the host's paid API key.

This is the security property the whole deployment rests on, so it is asserted
rather than assumed. Run it before every deploy:

    python tests/test_key_policy.py

No test framework required.
"""

import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import predict

FAKE_HOST_GROQ = "gsk_" + "H" * 40
FAKE_OPENAI = "sk-" + "O" * 40

failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


print("\n--- gated models must never reach the host key ---")

# Simulate the worst case: BOTH keys present in the environment, exactly as they
# would be if someone mistakenly configured OPENAI_API_KEY on the host.
hostile_env = {"GROQ_API_KEY": FAKE_HOST_GROQ, "OPENAI_API_KEY": FAKE_OPENAI}

with mock.patch.dict(os.environ, hostile_env, clear=True):
    with mock.patch.object(predict, "call_model") as called:
        status, body = predict.handle({"description": "a laptop", "model": "gpt-4.1-nano"})
        check("gpt-4.1-nano without BYOK is refused", status == 400, f"got {status}")
        check("...with key_required", body.get("error") == "key_required", str(body))
        check("...and no upstream call was made", not called.called)

    with mock.patch.object(predict, "call_model") as called:
        status, body = predict.handle({"description": "a laptop", "model": "gpt-oss-120b"})
        check("gpt-oss-120b without BYOK is refused", status == 400, f"got {status}")
        check("...and no upstream call was made", not called.called)

    # The decisive test: even holding OPENAI_API_KEY, the gated path must not find it.
    key, err = predict.resolve_key(predict.MODELS["gpt-4.1-nano"], byok="")
    check("resolve_key returns no key for gated model", key is None, str(key))
    check("resolve_key returns an error instead", err is not None)

    # A BYOK request must use the VISITOR's key, never the host's.
    with mock.patch.object(predict, "call_model", return_value=("$42", 0.1)) as called:
        visitor_key = "sk-" + "V" * 40
        status, body = predict.handle(
            {"description": "a laptop", "model": "gpt-4.1-nano", "api_key": visitor_key}
        )
        check("BYOK request succeeds", status == 200, str(body))
        used_key = called.call_args[0][1] if called.call_args else None
        check("...and used the visitor's key", used_key == visitor_key)
        check("...not the host's key", used_key != FAKE_OPENAI)

    # Free tier must ignore a supplied key and use the host's Groq key.
    with mock.patch.object(predict, "call_model", return_value=("$42", 0.1)) as called:
        status, body = predict.handle(
            {"description": "a laptop", "model": "gpt-oss-20b", "api_key": "sk-attacker"}
        )
        used_key = called.call_args[0][1] if called.call_args else None
        check("free tier uses the host Groq key", used_key == FAKE_HOST_GROQ, str(used_key))


print("\n--- wrong-provider keys are rejected ---")
with mock.patch.dict(os.environ, hostile_env, clear=True):
    with mock.patch.object(predict, "call_model") as called:
        status, body = predict.handle(
            {"description": "a laptop", "model": "gpt-4.1-nano", "api_key": FAKE_HOST_GROQ}
        )
        check("groq key rejected for an openai model", status == 400, f"got {status}")
        check("...without calling upstream", not called.called)


print("\n--- credentials never appear in error text ---")
leaky = f"Incorrect API key provided: {FAKE_OPENAI}. Also {FAKE_HOST_GROQ} is bad."
scrubbed = predict.scrub(leaky)
check("openai key scrubbed", FAKE_OPENAI not in scrubbed, scrubbed)
check("groq key scrubbed", FAKE_HOST_GROQ not in scrubbed, scrubbed)
check("redaction marker present", "[redacted]" in scrubbed)


print("\n--- malformed model replies never render as $0 ---")
check("empty reply -> None", predict.parse_price("") is None)
check("bare '$' -> None", predict.parse_price("$") is None)
check("'$42.50' -> 42.5", predict.parse_price("$42.50") == 42.5)
check("'$1,299' -> 1299", predict.parse_price("$1,299") == 1299.0)

with mock.patch.dict(os.environ, hostile_env, clear=True):
    with mock.patch.object(predict, "call_model", return_value=("$", 0.1)):
        status, body = predict.handle({"description": "a laptop", "model": "gpt-oss-20b"})
        check("truncated reply -> 502, not a $0 price", status == 502, str(body))
        check("...and no 'price' field is returned", "price" not in body)


print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("All key-policy checks passed.")
