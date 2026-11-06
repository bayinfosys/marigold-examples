"""
run_benchmark.py -- submit every parameter set to every model, record what
comes back.

The runner imposes no structure. It reads the model list from /models,
reads parameter sets from prompts.jsonl, and for each model sends every
set in turn, polls to completion, and flattens the whole response into a
CSV row. Grouping, repeats and sweeps live in the prompt file; analysis
groups on the label.

Serial by design: the worker holds one model and processes one message,
so concurrent submission would add queue wait to the durations being
measured. Models are processed one at a time for the same reason -- the
worker switches models on queue depth and interleaving would cost a full
reload between requests.

Results are written to JSONL as they arrive and converted to CSV at the
end. The CSV column set is the union of keys across all rows, which is
only known once the run finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests

MODEL_TYPE = "txt2img"
SUBMIT_PATH = "/gen/txt2img"
STATUS_PATH = "/output/gen/txt2img"

PENDING = {"queued", "processing"}


# ---------------------------------------------------------------------------
# submission core -- the part worth lifting into the CLI
# ---------------------------------------------------------------------------

def list_models(session, base_url: str, model_type: str) -> list[str]:
    """Active model names of one type, from Marigold's own catalogue route.

    /v1/models is the OpenAI-compatible surface and carries no type
    information; /models returns catalogue rows as a bare JSON array.
    """
    resp = session.get(f"{base_url}/models", timeout=30)
    resp.raise_for_status()

    entries = [m for m in resp.json() if m.get("type") == model_type]

    for m in entries:
        if not m.get("active", True):
            print(f"  skipping inactive model: {m['name']}")

    return [m["name"] for m in entries if m.get("active", True)]


def submit(session, base_url: str, body: dict) -> tuple[str, str]:
    """POST a request. Returns (message_id, poll_url).

    The Location header carries the poll URL when the API sets it; the
    fallback constructs it from the known route shape.
    """
    resp = session.post(f"{base_url}{SUBMIT_PATH}", json=body, timeout=30)
    resp.raise_for_status()

    message_id = resp.json()["message_id"]
    location = resp.headers.get("Location")

    poll_url = f"{base_url}{location}" if location else \
        f"{base_url}{STATUS_PATH}/{message_id}"

    return message_id, poll_url


def poll(session, poll_url: str, interval: float, timeout: float) -> dict:
    """Poll until terminal, or until timeout. Returns the envelope.

    The HTTP status describes the poll: 202 while pending, 200 once
    terminal, 404 for an unknown id. A job that failed is a successful
    poll returning status="error".
    """
    deadline = time.time() + timeout

    while True:
        resp = session.get(poll_url, timeout=30)

        if resp.status_code == 202:
            envelope = {"status": "queued"}
        else:
            resp.raise_for_status()
            envelope = resp.json()

        if envelope.get("status") not in PENDING:
            return envelope

        if time.time() >= deadline:
            return {"status": "timeout", "result": {
                "error": f"no terminal status within {timeout}s"}}

        time.sleep(interval)


# ---------------------------------------------------------------------------
# flattening
# ---------------------------------------------------------------------------

def flatten(obj, prefix: str = "") -> dict:
    """Flatten nested dicts and lists into dotted keys.

    Whatever the server reports ends up as a column. New metrics appear
    in the output without the runner knowing about them.
    """
    flat = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            flat.update(flatten(value, f"{prefix}{index}."))
    else:
        flat[prefix.rstrip(".")] = obj

    return flat


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_one(session, base_url, model, cell, interval, timeout) -> dict:
    body = dict(cell["request"], model=model)

    row = {"label": cell["label"], "model": model}
    row.update(flatten(body, "request."))

    started = time.time()

    try:
        message_id, poll_url = submit(session, base_url, body)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__} on submit: {exc}"
        return row

    row["message_id"] = message_id

    try:
        envelope = poll(session, poll_url, interval, timeout)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__} on poll: {exc}"
        return row

    row["client_seconds"] = round(time.time() - started, 3)
    row.update(flatten(envelope))

    return row


def write_csv(rows: list[dict], path: Path) -> None:
    """Union of keys across all rows, label and model first."""
    keys = sorted({k for row in rows for k in row})
    lead = [k for k in ("label", "model", "status") if k in keys]
    fieldnames = lead + [k for k in keys if k not in lead]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-id", required=True,
                        help="any string; Marigold tags requests by it")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompts", type=Path,
                        default=Path(__file__).parent / "prompts.jsonl")
    parser.add_argument("--models", nargs="+", default=None,
                        help="explicit model names, overriding the catalogue")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json",
                            "x-user-id": args.user_id})
    if args.api_key:
        session.headers["x-api-key"] = args.api_key

    with args.prompts.open() as f:
        cells = [json.loads(line) for line in f if line.strip()]

    models = args.models or list_models(session, base_url, MODEL_TYPE)

    if not models:
        raise SystemExit(f"no active {MODEL_TYPE} models at {base_url}")

    total = len(models) * len(cells)
    print(f"{len(models)} models, {len(cells)} parameter sets, {total} requests")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out.with_suffix(".jsonl")

    rows = []
    done = 0

    with jsonl_path.open("w") as raw:
        for model in models:
            print(f"\n{model}")

            for cell in cells:
                row = run_one(session, base_url, model, cell,
                              args.poll_interval, args.timeout)

                raw.write(json.dumps(row) + "\n")
                raw.flush()
                rows.append(row)

                done += 1
                status = row.get("status", row.get("error", "?"))
                print(f"  [{done}/{total}] {cell['label']}: {status}")

    write_csv(rows, args.out)

    complete = sum(1 for r in rows if r.get("status") == "complete")
    print(f"\n{complete}/{total} complete")
    print(f"  {args.out}")
    print(f"  {jsonl_path}")


if __name__ == "__main__":
    main()
