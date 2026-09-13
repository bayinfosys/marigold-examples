"""
build_prompts.py -- parameter sets for the image generation benchmark.

Each line of prompts.jsonl is:

    {"label": "steps_004_r0", "request": {...}}

`request` is sent as the POST body verbatim, with `model` added by the
runner. `label` is recorded in the CSV and never sent.

Repeats are expressed here rather than in the runner. message_id is a
hash of the request body, so two identical bodies are the same job and
the second returns the first from cache. Each repeat therefore carries a
distinct `nonce`, a field ModelRequest provides for exactly this.

Sweeps vary one parameter at a time. Prompt text is held constant across
them: the same model at the same steps and resolution executes the same
graph with the same tensor shapes whatever the words are, so varying the
text adds rows without adding information about power or memory.
"""

import json
from pathlib import Path

SEED = 20260910
REPEATS = 5

PROMPT = (
    "a stone castle on a green hill. photographic, natural light, "
    "shallow depth of field. No text, no labels, no borders."
)

BASELINE_STEPS = 25
BASELINE_GUIDANCE = 7.5

STEP_VALUES = [4, 8, 16, 25, 40, 200]
GUIDANCE_VALUES = [0.0, 1.0, 7.5, 15.0]


def _cell(label: str, **overrides) -> dict:
    request = {
        "prompt": PROMPT,
        "num_inference_steps": BASELINE_STEPS,
        "guidance_scale": BASELINE_GUIDANCE,
        "seed": SEED,
    }
    request.update(overrides)
    return {"label": label, "request": request}


def build_cells() -> list[dict]:
    cells = [
        _cell(f"steps_{steps:03d}", num_inference_steps=steps)
        for steps in STEP_VALUES
    ]

    cells += [
        _cell(f"guidance_{g:04.1f}".replace(".", "p"), guidance_scale=g)
        for g in GUIDANCE_VALUES
    ]

    return cells


def build_prompt_set() -> list[dict]:
    out = []

    for repeat in range(REPEATS):
        for cell in build_cells():
            request = dict(cell["request"])
            request["nonce"] = f"r{repeat}"
            out.append({
                "label": f"{cell['label']}_r{repeat}",
                "request": request,
            })

    return out


if __name__ == "__main__":
    out_path = Path(__file__).parent / "prompts.jsonl"
    prompt_set = build_prompt_set()

    with out_path.open("w") as f:
        for p in prompt_set:
            f.write(json.dumps(p) + "\n")

    print(f"wrote {len(prompt_set)} parameter sets to {out_path}")
