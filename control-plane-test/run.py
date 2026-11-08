"""
quick_platform_test.py -- "is it up?"

A small-scale platform test, not a benchmark: a handful of tiny
models across text-embedding, image-embedding, and instruct, each
exercised once, answering "does a fresh deployment actually work end
to end" rather than measuring throughput or quality. Use
run_benchmark.py for the real thing once this passes.

Checks, in order:
    1. GET /models           -- catalogue loaded from models.yaml
    2. POST /embed/text       )
       POST /embed/image      )  one submit+poll per catalogue model,
       POST /gen/instruct     )  dispatched on the model's declared type
    3. GET  /output/...      -- poll until complete, print status transitions

Scope: model listing, embed, and gen/instruct only, deliberately -- no
/users/ routes, no /eval/, no openai-compat /v1/models (that gets its
own tester later). No distinction is made between plain instruct
models and vision-capable ones -- every instruct model gets the same
text-only prompt. If a model needs an image to answer meaningfully
that's a separate, deliberate test, not something this script infers
from the model name.

Exit code 0 on success, 1 on any failure, so this can be scripted or
run in CI without reading output.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time

import requests

import base64

# A small solid-colour test JPEG, generated once and embedded here so
# image-embedding/img2txt checks have zero external dependency by
# default. httpbin.org (the previous default) is a free public service
# with no uptime guarantee -- a network blip there would fail this test
# for reasons unrelated to Marigold itself. --image-url still exists
# for testing against a real photo when that's what you actually want.
_DEFAULT_IMAGE_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDtaKKK/KD6MKKr32oWmm2xub24jgiH8TnGTgnA9TgHgc1x9/8AE+whk2WNjNdAEhnkcRA46EcEkH3Arqw+CxGI/hRb/L73oZzqwh8TO4org7X4p2ryEXelzRR44aKUSHPpgheOveuw03V9P1iEzafdxzqPvBThl5I5U8joeoqsRgcTh1erBpfevvQoVqc/hZcooorjNQqvqF9BpthPe3LYigQu3IyfYZ7k8D3NWK4f4n37w6XZ2KbgLmRndg2AQgHykdxlgf8AgIrqwWH+sYiNLu/w3f4GdWfJByOI8QeILvxDfm5uTsjXIhhBysa/1J7nv9MAZdFFfpVOnClBQgrJHgyk5O7Crmlard6Lfpe2UmyReCDyrr3Vh3B/zzVOinOEZxcZK6YJtO6PctC1iDXdJiv4F2b8h4ywJjYdQcfmPYg45rRrzf4XX7re3unHcY3jEy/NwpUhTx6ncOf9kV6RX5xmOGWGxMqa26ejPcoVPaU1JhXB/FO1d7LT7sFfLikeNh3ywBH4fIf0rvKp6vpsWsaTc6fMcLOmA3Pyt1U9RnBAOPapwOIWHxMKr2T/AAejHWhz03E8Ioq5qulXei372V7HskXkEcq69mU9wf8APNU6/SYTjOKlF3TPCaadmFFFTWdncX93HaWkTSzSnaiL1P8A9b37U21FXewkrnY/C61d9WvbsFfLigEbDvlmBH4fIf0r0ysjwxoY8P6LHZFleYkyTOucM59M9gAB2zjOOa16/OczxMcTipTjtsvke7QpunTSYUUUV55sZ2saFp+u2wgv4d+zJjdThoyRjIP9DxwMjiuIv/hddLJnTtQhkjJPy3AKFR2GQDuPrwK9Ioruw2Y4nDK1OWnbdGNShTqayR5na/C7UXkIu9Qtoo8cNEGkOfTBC8de9drofhjS/D6sbKJjM42vNI252Gc49APoBnAznFa9FViczxWJjyzlp2WgU6FOm7pBRRRXnmx//9k="


def build_headers(args) -> dict:
    h = {"Content-Type": "application/json", "x-user-id": args.user_id}
    if args.api_key:
        h["x-api-key"] = args.api_key
    return h


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def get_models(base_url: str, headers: dict) -> list[dict] | None:
    """GET /models -- catalogue, returning name+type so callers can
    dispatch on type rather than guessing from the model name."""
    try:
        resp = requests.get(f"{base_url}/models", headers=headers, timeout=10)
    except requests.exceptions.RequestException as exc:
        check("GET /models", False, f"{type(exc).__name__}: {exc}")
        return None

    ok = resp.ok and len(resp.json()) > 0
    check("GET /models", ok, f"{resp.status_code}, {len(resp.json()) if resp.ok else '?'} models" if resp.ok else resp.text[:300])
    return resp.json() if ok else None


def download_image_b64(url: str) -> str:
    """Fetch an image and return it base64-encoded. Only called when
    --image-url is explicitly passed -- the default path uses
    _DEFAULT_IMAGE_B64 and never touches the network for this."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("ascii")


def instruct_payload(model: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 16,
        "temperature": 1.0,
    }


def embed_payload(model: str, content: str) -> dict:
    """Same field for text and image -- input, base64-encoded for
    images. See EmbedTextRequest/EmbedImageRequest in api/models.py."""
    return {"model": model, "input": content}


def img2txt_payload(model: str, image_b64: str) -> dict:
    """See Img2TxtRequest / img2txt.py -- input is base64 image, prompt
    is optional and defaults server-side to 'Describe this image.'."""
    return {"model": model, "input": image_b64, "max_tokens": 64}


def submit_and_poll(
    base_url: str, headers: dict, mode: str, task: str, model: str, payload: dict, timeout_s: float
) -> bool:
    """POST /{mode}/{task} then poll /output/{mode}/{task}/{message_id}.

    One poll loop for every model type -- output.py's poll route is
    the same shape regardless of what was submitted.
    """
    label = f"{mode}/{task} ({model})"

    try:
        submit_resp = requests.post(f"{base_url}/{mode}/{task}", headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as exc:
        return check(f"POST /{mode}/{task} ({model})", False, f"{type(exc).__name__}: {exc}")

    if not submit_resp.ok:
        return check(f"POST /{mode}/{task} ({model})", False, f"{submit_resp.status_code}: {submit_resp.text[:300]}")

    message_id = submit_resp.json()["message_id"]
    check(f"POST /{mode}/{task} ({model})", True, f"message_id={message_id}")

    last_status = None
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            poll_resp = requests.get(f"{base_url}/output/{mode}/{task}/{message_id}", headers=headers, timeout=10)
        except requests.exceptions.RequestException as exc:
            return check(f"GET /output/{mode}/{task}/...", False, f"{type(exc).__name__}: {exc}")

        if poll_resp.status_code == 202:
            status = "queued"
        elif poll_resp.ok:
            status = poll_resp.json().get("status")
        else:
            return check(f"GET /output/{mode}/{task}/...", False, f"{poll_resp.status_code}: {poll_resp.text[:300]}")

        if status != last_status:
            print(f"    status -> {status}  ({time.monotonic() - start:.1f}s)")
            last_status = status

        if status == "complete":
            return check(f"job complete ({label})", True, f"result: {str(poll_resp.json()['result'])[:200]}")
        if status == "error":
            return check(f"job complete ({label})", False, f"job errored: {poll_resp.text[:300]}")

        time.sleep(1.0)

    return check(f"job complete ({label})", False, f"timed out after {timeout_s}s, last status={last_status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--user-id", type=str, default="quick-platform-test")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait per job")
    parser.add_argument(
        "--image-url", type=str, default=None,
        help="image fetched once and reused for any image-embedding checks",
    )
    args = parser.parse_args()

    headers = build_headers(args)
    results = []

    models = get_models(args.base_url, headers)
    results.append(models is not None)

    if models:
        image_b64 = None  # fetched once, lazily, only if something needs it

        for entry in models:
            name, mtype = entry["name"], entry["type"]

            if mtype == "text-embedding":
                results.append(submit_and_poll(
                    args.base_url, headers, "embed", "text", name,
                    embed_payload(name, "a quick platform test sentence"), args.timeout,
                ))

            elif mtype == "image-embedding":
                if image_b64 is None:
                    image_b64 = download_image_b64(args.image_url) if args.image_url else _DEFAULT_IMAGE_B64
                results.append(submit_and_poll(
                    args.base_url, headers, "embed", "image", name,
                    embed_payload(name, image_b64), args.timeout,
                ))

            elif mtype == "instruct":
                results.append(submit_and_poll(
                    args.base_url, headers, "gen", "instruct", name,
                    instruct_payload(name), args.timeout,
                ))

            elif mtype == "img2txt":
                if image_b64 is None:
                    image_b64 = download_image_b64(args.image_url) if args.image_url else _DEFAULT_IMAGE_B64
                results.append(submit_and_poll(
                    args.base_url, headers, "gen", "img2txt", name,
                    img2txt_payload(name, image_b64), args.timeout,
                ))

            else:
                print(f"[SKIP] {name} -- no payload builder for type={mtype!r}")

    else:
        print("[SKIP] submit/poll -- no models available")
        results.append(False)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
