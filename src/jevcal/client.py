"""Minimal TypeSafe Jev client: stdlib only, retries on rate limits and 5xx."""

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"  # "jev-1.13" is rejected; every row records the build the response reports
ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_key() -> str:
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key.strip()
    env = (ROOT / ".env").read_text()
    pairs = dict(line.split("=", 1) for line in env.splitlines() if "=" in line)
    return pairs["TYPESAFE_API_KEY"].strip()


def ask(key: str, state: str, questions: dict, model: str = MODEL) -> dict:
    """POST one state with a map of typed questions; return the parsed response."""
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")
