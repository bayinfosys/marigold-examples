"""
run_benchmark.py -- one pass, re-run whenever you like.

message_id is a deterministic hash of the request body, and every field
in that body (model, prompt, steps, frames, resolution, seed) is fixed by
the prompt set. So resubmitting already-completed work is always a cheap
cache-status check, never a re-run, with no nonce needed to make the
bodies distinct.

A model that fails to load fails for every prompt, and the worker drains
its whole queue on a load failure. So each model gets one canary prompt
first: if that comes back a load failure, the remaining prompts are
written as skipped rows rather than submitted, which would cost a full
model load attempt per batch for identical results.

Binary output retrieval returns 501, so this records the output
reference -- path and mimetype -- rather than fetching bytes. The files
are under MARIGOLD_CACHE_DIR/data/outputs/txt2vid on the host running
the worker.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


MODEL_TYPE = "txt2vid"
SUBMIT_PATH = "/gen/txt2vid"
STATUS_PATH = "/output/gen/txt2vid"

# Substrings in a job's error text that mean the model itself could not be
# loaded, rather than this particular request failing.
LOAD_FAILURE_MARKERS = ("failed VRAM check", "model load failed", "unknown model_type")

FIELDNAMES = [
    "model", "prompt_id", "prompt_group", "success",
    "prompt", "num_inference_steps", "num_frames", "fps", "width", "height", "seed",
    "output_path", "output_mimetype",
    "http_status", "error", "raw_response",
    "server_duration_seconds", "server_inference_seconds", "server_load_seconds",
    "vram_usage_bytes", "vram_usage_bytes_peak",
    "model_size_bytes", "process_memory_kb",
    "power_watts_peak", "power_watts_mean", "cpu_offload_bytes",
]


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def load_config(args) -> dict:
    cfg = json.loads(args.config.read_text()) if args.config else {}
    if args.base_url: cfg["base_url"] = args.base_url
    if args.api_key: cfg["api_key"] = args.api_key
    if args.user_id: cfg["user_id"] = args.user_id
    missing = [k for k in ("base_url", "user_id") if k not in cfg]
    if missing:
        raise SystemExit(f"missing required config values: {missing}")
    cfg.setdefault("api_key", None)
    return cfg


def list_models(base_url: str, headers: dict, model_type: str) -> list[str]:
    """Model names from Marigold's own /models endpoint, filtered by type.

    /v1/models is the OpenAI-compatible surface open-webui consumes and
    carries no type information. /models returns the catalogue rows.

    Inactive entries are skipped: a model the worker deactivated after a
    load failure would queue forever, or be rejected at submission.
    """
    resp = requests.get(f"{base_url}/models", headers=headers, timeout=30)
    resp.raise_for_status()

    entries = [m for m in resp.json() if m.get("type") == model_type]

    inactive = [m["name"] for m in entries if not m.get("active", True)]
    if inactive:
        print(f"skipping inactive models: {', '.join(inactive)}")

    return [m["name"].lower() for m in entries if m.get("active", True)]


def build_payload(model: str, prompt: dict) -> dict:
    payload = {
        "model": model,
        "prompt": prompt["prompt"],
        "num_inference_steps": prompt["num_inference_steps"],
        "num_frames": prompt["num_frames"],
        "fps": prompt["fps"],
        "width": prompt["width"],
        "height": prompt["height"],
        "seed": prompt["seed"],
    }

    # negative_prompt is passed through to the pipeline unconditionally by
    # Txt2VidModel._run, and some pipelines reject the kwarg. Only send it
    # when it carries something.
    if prompt.get("negative_prompt"):
        payload["negative_prompt"] = prompt["negative_prompt"]

    return payload


def _with_retries(fn, attempts=3, backoff=1.0):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
    raise last_exc


def _job_error(data: dict) -> str:
    """The server's own error text for a failed job.

    handle_status returns 200 with status='error' and
    result={'error': '...'} for a job the worker failed, including one
    drained after a load failure.
    """
    result = data.get("result") or {}
    return str(result.get("error", "")) or f"job status={data.get('status')}"


def _is_load_failure(row: dict) -> bool:
    """Whether this row's failure applies to the whole model."""
    blob = f"{row.get('error', '')} {row.get('raw_response', '')}"
    return any(marker in blob for marker in LOAD_FAILURE_MARKERS)


def _base_row(model: str, prompt: dict) -> dict:
    return {
        "model": model,
        "prompt_id": prompt["prompt_id"],
        "prompt_group": prompt["prompt_group"],
        "prompt": prompt["prompt"],
        "num_inference_steps": prompt["num_inference_steps"],
        "num_frames": prompt["num_frames"],
        "fps": prompt["fps"],
        "width": prompt["width"],
        "height": prompt["height"],
        "seed": prompt["seed"],
    }


def _skipped_row(model: str, prompt: dict, reason: str) -> dict:
    """A row for a prompt never submitted, so the CSV shows the model was
    attempted rather than silently omitting it."""
    row = _base_row(model, prompt)
    row.update(success=False, http_status=None,
               error=f"skipped: {reason}", raw_response="")
    return row


def _extract(row: dict, data: dict) -> dict:
    """Pull usage metrics and the output reference out of a completed job."""
    result = data["result"]
    usage = result["usage"]

    outputs = result.get("outputs") or {}
    reference = next(iter(outputs.values()), {}) if outputs else {}

    row.update(
        success=True, error="", raw_response="",
        output_path=reference.get("path", ""),
        output_mimetype=reference.get("mimetype", ""),
        server_duration_seconds=usage["duration"] / 1000.0,
        server_inference_seconds=usage["inference"] / 1000.0,
        server_load_seconds=usage["load_time_ms"] / 1000.0,
        vram_usage_bytes=usage["vram_usage_bytes"],
        vram_usage_bytes_peak=usage["vram_usage_bytes_peak"],
        model_size_bytes=usage["model_size_bytes"],
        process_memory_kb=usage["memory_usage"],
        power_watts_peak=usage["power_watts_peak"],
        power_watts_mean=usage["power_watts_mean"],
        cpu_offload_bytes=usage["cpu_offload_bytes"],
    )
    return row


def run_one(session, base_url, headers, model, prompt, wait_seconds, poll_interval) -> dict:
    row = _base_row(model, prompt)
    payload = build_payload(model, prompt)

    try:
        submit_resp = _with_retries(
            lambda: session.post(f"{base_url}{SUBMIT_PATH}", headers=headers,
                                 json=payload, timeout=30)
        )
    except Exception as exc:
        row.update(success=False, http_status=None,
                   error=f"{type(exc).__name__} on submit: {exc}", raw_response="")
        return row

    if not submit_resp.ok:
        row.update(success=False, http_status=submit_resp.status_code,
                   error=f"submit HTTP {submit_resp.status_code}",
                   raw_response=submit_resp.text[:2000])
        return row

    message_id = submit_resp.json()["message_id"]
    deadline = time.time() + wait_seconds

    while True:
        try:
            poll_resp = _with_retries(
                lambda: session.get(f"{base_url}{STATUS_PATH}/{message_id}",
                                    headers=headers, timeout=30)
            )
        except Exception as exc:
            row.update(success=False, http_status=None,
                       error=f"{type(exc).__name__} checking status: {exc}",
                       raw_response="")
            return row

        row["http_status"] = poll_resp.status_code

        if not poll_resp.ok and poll_resp.status_code != 202:
            row.update(success=False,
                       error=f"status check HTTP {poll_resp.status_code}",
                       raw_response=poll_resp.text[:2000])
            return row

        data = poll_resp.json() if poll_resp.status_code != 202 else {}
        status = data.get("status") if data else "queued"

        if status == "complete":
            try:
                return _extract(row, data)
            except (KeyError, IndexError) as exc:
                row.update(success=False,
                           error=f"{type(exc).__name__} extracting fields: {exc}",
                           raw_response=str(data)[:2000])
                return row

        if status in ("failed", "error"):
            row.update(success=False, error=_job_error(data),
                       raw_response=str(data)[:2000])
            return row

        if time.time() >= deadline:
            row.update(success=False, error="still queued -- rerun later",
                       raw_response="")
            return row

        time.sleep(poll_interval)


def sample_prompts(prompts: list[dict], max_prompts: int | None) -> list[dict]:
    """Stratified sample across prompt_group, not the first N as they
    appear in the file -- build_prompt_set() orders baseline first, so a
    plain slice would never touch the sweeps at all, defeating the point
    of a smoke test."""
    if max_prompts is None:
        return prompts

    by_group: dict[str, list[dict]] = {}
    for p in prompts:
        by_group.setdefault(p["prompt_group"], []).append(p)

    groups = list(by_group.values())
    sampled: list[dict] = []
    i = 0
    while len(sampled) < max_prompts and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(sampled) < max_prompts:
                sampled.append(g[i])
        i += 1
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--prompts", type=Path,
                        default=Path(__file__).parent / "prompts.jsonl")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="explicit model names, overriding the /models catalogue")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="a single clip saturates the GPU; higher values add polling traffic only")
    parser.add_argument("--wait-seconds", type=int, default=3600,
                        help="how long to wait for one job before recording it as queued")
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument(
        "--max-prompts", type=int, default=None,
        help="Stratified sample of N prompts per model, spread across prompt groups, for a smoke test.",
    )
    args = parser.parse_args()

    cfg = load_config(args)
    base_url = cfg["base_url"].rstrip("/")
    headers = {"Content-Type": "application/json", "x-user-id": cfg["user_id"]}
    if cfg["api_key"]:
        headers["x-api-key"] = cfg["api_key"]

    prompts = sample_prompts(load_prompts(args.prompts), args.max_prompts)
    models = args.models or list_models(base_url, headers, MODEL_TYPE)

    if not models:
        raise SystemExit(f"no active {MODEL_TYPE} models at {base_url}")

    total = len(models) * len(prompts)
    print(f"{len(models)} {MODEL_TYPE} models, {len(prompts)} prompts, {total} calls total")

    session = requests.Session()
    session.mount(base_url, requests.adapters.HTTPAdapter(
        pool_maxsize=args.workers, pool_connections=args.workers))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        # Grouped by model rather than interleaved: the worker holds one
        # model at a time and switches on queue depth, so interleaving
        # buys a full reload between every request.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            done, complete, queued, skipped = 0, 0, 0, 0

            for model in models:
                # Canary. A load failure applies to every prompt for this
                # model, and the worker drains its queue on one, so
                # submitting the rest costs a full load attempt per batch
                # for identical rows.
                canary = run_one(session, base_url, headers, model,
                                 prompts[0], args.wait_seconds, args.poll_interval)
                writer.writerow(canary)
                f.flush()
                done += 1

                if canary["success"] is True:
                    complete += 1
                elif canary.get("error") == "still queued -- rerun later":
                    queued += 1

                if _is_load_failure(canary):
                    reason = canary["error"]
                    print(f"  {model}: {reason}")
                    print(f"  skipping {len(prompts) - 1} remaining prompt(s) for this model")
                    for prompt in prompts[1:]:
                        writer.writerow(_skipped_row(model, prompt, reason))
                        done += 1
                        skipped += 1
                    f.flush()
                    continue

                futures = [
                    pool.submit(run_one, session, base_url, headers, model,
                                prompt, args.wait_seconds, args.poll_interval)
                    for prompt in prompts[1:]
                ]

                for future in as_completed(futures):
                    row = future.result()
                    writer.writerow(row)
                    f.flush()
                    done += 1
                    if row["success"] is True:
                        complete += 1
                    elif row.get("error") == "still queued -- rerun later":
                        queued += 1
                    if done % 5 == 0:
                        print(f"  {done}/{total} ({complete} complete, {queued} still queued)")

    failed = done - complete - queued - skipped
    print(f"done. {complete} complete, {queued} still queued, {skipped} skipped, "
          f"{failed} failed -- rerun this script later to pick up more completions.")


if __name__ == "__main__":
    main()
