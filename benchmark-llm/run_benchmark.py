"""
run_benchmark.py -- one pass, re-run whenever you like.

message_id is a deterministic hash of the request body, so resubmitting
already-completed work is always a cheap cache-status check, never a
re-run. That means there's no separate "already done" tracking needed --
just resubmit everything, check once, and write a fresh CSV. Run this
again in an hour and it picks up whatever's finished since.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

FIELDNAMES = [
    "model", "prompt_id", "prompt_group", "success",
    "requested_max_tokens",
    "completion_tokens", "prompt_tokens", "response_text",
    "http_status", "error", "raw_response",
    "server_duration_seconds", "server_inference_seconds", "server_load_seconds",
    "vram_usage_bytes", "vram_usage_bytes_peak",
    "model_size_bytes", "process_memory_kb",
    "power_watts_peak", "power_watts_mean", "cpu_offload_bytes",
]

DEFAULT_MAX_TOKENS=2048


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
    for key in ("base_url", "api_key", "user_id"):
        val = getattr(args, key.replace("base_url", "base_url"))
    if args.base_url: cfg["base_url"] = args.base_url
    if args.api_key: cfg["api_key"] = args.api_key
    if args.user_id: cfg["user_id"] = args.user_id
    missing = [k for k in ("base_url", "user_id") if k not in cfg]
    if missing:
        raise SystemExit(f"missing required config values: {missing}")
    cfg.setdefault("api_key", None)
    return cfg


def list_instruct_models(base_url: str, api_key: Optional[str], user_id: str) -> list[str]:
    headers = {"x-user-id": user_id}
    if api_key:
        headers["x-api-key"] = api_key
    resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


def nonce_for(model: str, prompt_id: str) -> str:
    digest = hashlib.sha256(f"{model}:{prompt_id}".encode("utf-8")).hexdigest()
    return f"0x{digest[:8]}"


def build_payload(model: str, prompt: dict, max_tokens: int, temperature: float) -> dict:
    messages = (
        [{"role": "system", "content": prompt["system"]}] if prompt["system"] else []
    ) + [{"role": "user", "content": prompt["user"]}]
    payload = {
        "model": model, "nonce": nonce_for(model, prompt["prompt_id"]),
        "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
    }
    if prompt.get("json_schema") is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": prompt["json_schema"], "strict": True},
        }
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


def run_one(session, base_url, headers, model, prompt, max_tokens, temperature) -> dict:
    row = {
        "model": model, "prompt_id": prompt["prompt_id"], "prompt_group": prompt["prompt_group"],
    }
    max_tokens = prompt.get("max_tokens", DEFAULT_MAX_TOKENS)
    payload = build_payload(model, prompt, max_tokens, temperature)
    row["requested_max_tokens"] = max_tokens

    try:
        submit_resp = _with_retries(lambda: session.post(f"{base_url}/gen/instruct", headers=headers, json=payload, timeout=30))
    except Exception as exc:
        row.update(success=False, http_status=None, error=f"{type(exc).__name__} on submit: {exc}", raw_response="")
        return row
    if not submit_resp.ok:
        row.update(success=False, http_status=submit_resp.status_code, error=f"submit HTTP {submit_resp.status_code}", raw_response=submit_resp.text[:2000])
        return row

    message_id = submit_resp.json()["message_id"]

    try:
        poll_resp = _with_retries(lambda: session.get(f"{base_url}/output/gen/instruct/{message_id}", headers=headers, timeout=30))
    except Exception as exc:
        row.update(success=False, http_status=None, error=f"{type(exc).__name__} checking status: {exc}", raw_response="")
        return row

    row["http_status"] = poll_resp.status_code
    if poll_resp.status_code == 202:
        row.update(success=False, error="still queued -- rerun later", raw_response="")
        return row
    if not poll_resp.ok:
        row.update(success=False, error=f"status check HTTP {poll_resp.status_code}", raw_response=poll_resp.text[:2000])
        return row

    data = poll_resp.json()
    status = data.get("status")
    if status != "complete":
        row.update(success=False, error=f"job status={status}", raw_response=str(data)[:2000])
        return row

    try:
        result = data["result"]
        usage = result["usage"]
        row.update(
            success=True, error="", raw_response="",
            completion_tokens=usage["output_tokens"], prompt_tokens=usage["input_tokens"],
            response_text=result["choices"][0]["content"],
            server_duration_seconds=usage["duration"] / 1000.0,
            server_inference_seconds=usage["inference"] / 1000.0,
            server_load_seconds=usage["load_time_ms"] / 1000.0,
            vram_usage_bytes=usage["vram_usage_bytes"], vram_usage_bytes_peak=usage["vram_usage_bytes_peak"],
            model_size_bytes=usage["model_size_bytes"], process_memory_kb=usage["memory_usage"],
            power_watts_peak=usage["power_watts_peak"], power_watts_mean=usage["power_watts_mean"],
            cpu_offload_bytes=usage["cpu_offload_bytes"],
        )
    except (KeyError, IndexError) as exc:
        row.update(success=False, error=f"{type(exc).__name__} extracting fields: {exc}", raw_response=str(data)[:2000])

    return row


def sample_prompts(prompts: list[dict], max_prompts: int | None) -> list[dict]:
    """Stratified sample across prompt_group, not the first N as they
    appear in the file -- build_prompt_set() orders varying_context
    first, so a plain slice would never touch structured or long_form
    prompts at all, defeating the point of a smoke test."""
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=16)
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

    prompts = load_prompts(args.prompts)
    prompts = sample_prompts(prompts, args.max_prompts)
    models = list_instruct_models(base_url, cfg["api_key"], cfg["user_id"])
    total = len(models) * len(prompts)
    print(f"{len(models)} instruct models, {len(prompts)} prompts, {total} calls total")

    session = requests.Session()
    session.mount(base_url, requests.adapters.HTTPAdapter(pool_maxsize=args.workers, pool_connections=args.workers))

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_one, session, base_url, headers, model, prompt, args.max_tokens, args.temperature)
                for model in models for prompt in prompts
            ]
            done, complete, queued = 0, 0, 0
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                f.flush()
                done += 1
                if row["success"] == True: complete += 1
                elif row.get("error") == "still queued -- rerun later": queued += 1
                if done % 500 == 0:
                    print(f"  {done}/{total} ({complete} complete, {queued} still queued)")

    print(f"done. {complete} complete, {queued} still queued, {done - complete - queued} failed -- rerun this script later to pick up more completions.")


if __name__ == "__main__":
    main()
