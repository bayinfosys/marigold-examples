"""
run_benchmark.py -- one pass, re-run whenever you like.

message_id is a deterministic hash of the request body, and every field
in that body (model, prompt, steps, resolution, seed) is fixed by the
prompt set. So resubmitting already-completed work is always a cheap
cache-status check, never a re-run, with no nonce needed to make the
bodies distinct.

Binary output retrieval returns 501, so this records the output
reference -- path and mimetype -- rather than fetching bytes. The files
are under MARIGOLD_CACHE_DIR/data/outputs/txt2img on the host running
the worker.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

MODEL_TYPE = "txt2img"
SUBMIT_PATH = "/gen/txt2img"
STATUS_PATH = "/output/gen/txt2img"

FIELDNAMES = [
    "model", "prompt_id", "prompt_group", "success",
    "prompt", "num_inference_steps", "width", "height", "seed",
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
    carries no type information. /models is Marigold's own view: type,
    cached state, and everything else in the catalogue.
    """
    resp = requests.get(f"{base_url}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    entries = [m for m in data.get("models", []) if m.get("type") == model_type]

    uncached = [m["name"] for m in entries if m.get("cached") is False]
    if uncached:
        print(f"warning: not cached, will fail at load: {', '.join(uncached)}")

    return [m["name"].lower() for m in entries]


def build_payload(model: str, prompt: dict) -> dict:
    payload = {
        "model": model,
        "prompt": prompt["prompt"],
        "num_inference_steps": prompt["num_inference_steps"],
        "width": prompt["width"],
        "height": prompt["height"],
        "seed": prompt["seed"],
    }

    # negative_prompt is passed through to the pipeline unconditionally
    # by Txt2ImgModel._run, and some DiT pipelines reject the kwarg.
    # Only send it when it carries something.
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
    row = {
        "model": model,
        "prompt_id": prompt["prompt_id"],
        "prompt_group": prompt["prompt_group"],
        "prompt": prompt["prompt"],
        "num_inference_steps": prompt["num_inference_steps"],
        "width": prompt["width"],
        "height": prompt["height"],
        "seed": prompt["seed"],
    }

    payload = build_payload(model, prompt)

    try:
        submit_resp = _with_retries(
            lambda: session.post(f"{base_url}{SUBMIT_PATH}", headers=headers, json=payload, timeout=30)
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
                lambda: session.get(f"{base_url}{STATUS_PATH}/{message_id}", headers=headers, timeout=30)
            )
        except Exception as exc:
            row.update(success=False, http_status=None,
                       error=f"{type(exc).__name__} checking status: {exc}", raw_response="")
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
            row.update(success=False, error=f"job status={status}",
                       raw_response=str(data)[:2000])
            return row

        if time.time() >= deadline:
            row.update(success=False, error="still queued -- rerun later", raw_response="")
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
    parser.add_argument("--prompts", type=Path, default=Path(__file__).parent / "prompts.jsonl")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="explicit model names")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-seconds", type=int, default=900,
                        help="how long to wait for one job before recording it as queued")
    parser.add_argument("--poll-interval", type=float, default=5.0)
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
    models = args.models or list_models(args.models_yaml, MODEL_TYPE)

    if not models:
        raise SystemExit(f"no {MODEL_TYPE} models found in {args.models_yaml}")

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
            done, complete, queued = 0, 0, 0

            for model in models:
                futures = [
                    pool.submit(run_one, session, base_url, headers, model,
                                prompt, args.wait_seconds, args.poll_interval)
                    for prompt in prompts
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
                    if done % 25 == 0:
                        print(f"  {done}/{total} ({complete} complete, {queued} still queued)")

    print(f"done. {complete} complete, {queued} still queued, "
          f"{done - complete - queued} failed -- rerun this script later to pick up more completions.")


if __name__ == "__main__":
    main()
