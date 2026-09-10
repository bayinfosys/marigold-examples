"""
build_prompts.py -- prompt set for the Marigold image generation benchmark.

Three groups, generated combinatorially (subjects x styles) rather than
hand-typed -- growing the set means editing the SUBJECTS/STYLES lists.

Image generation cost is dominated by step count and pixel count, with
prompt length a distant third. The groups vary one axis each:

  baseline          -- fixed steps and resolution, varied subject and
                       style. The cell every other group is compared to.

  step_sweep        -- one prompt, step count varied. Measures the
                       linear term. Distilled models (schnell, turbo)
                       are included deliberately: the cost curve is real
                       even where the quality curve is flat.

  resolution_sweep  -- one prompt, resolution varied. Measures the
                       quadratic term and finds the VRAM ceiling.

Every prompt carries a fixed seed so a re-run reproduces the same image
and the same message_id, making resubmission a cache-status check rather
than a re-run.
"""

import json
from pathlib import Path

SEED = 20260910

SUBJECTS = [
    "a stone castle on a green hill",
    "a fishing boat at a harbour wall",
    "a red fox in long grass",
    "a terraced street in the rain",
    "a lighthouse on a rocky headland",
    "a market stall stacked with fruit",
    "a steam locomotive at a platform",
    "an oak tree alone in a field",
    "a canal lock with wooden gates",
    "a slate quarry under low cloud",
]

STYLES = [
    "photographic, natural light, shallow depth of field",
    "a children's crayon drawing, waxy texture, visible paper grain",
    "an oil painting, thick impasto, muted palette",
    "a technical line drawing, clean strokes, no shading",
]

BASELINE_STEPS = 25
BASELINE_SIZE = (1024, 1024)

STEP_VALUES = [4, 8, 16, 25, 40]

RESOLUTION_VALUES = [
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1024, 1536),
    (1536, 1536),
]

SWEEP_SUBJECT = SUBJECTS[0]
SWEEP_STYLE = STYLES[0]

NEGATIVE = ""


def _prompt(subject: str, style: str) -> str:
    return f"{subject}. {style}. No text, no labels, no borders."


def build_baseline() -> list[dict]:
    prompts, idx = [], 0
    width, height = BASELINE_SIZE
    for subject in SUBJECTS:
        for style in STYLES:
            prompts.append({
                "prompt_id": f"baseline_{idx:03d}",
                "prompt_group": "baseline",
                "prompt": _prompt(subject, style),
                "negative_prompt": NEGATIVE,
                "num_inference_steps": BASELINE_STEPS,
                "width": width,
                "height": height,
                "seed": SEED,
            })
            idx += 1
    return prompts


def build_step_sweep() -> list[dict]:
    prompts = []
    width, height = BASELINE_SIZE
    for idx, steps in enumerate(STEP_VALUES):
        prompts.append({
            "prompt_id": f"step_sweep_{steps:03d}",
            "prompt_group": "step_sweep",
            "prompt": _prompt(SWEEP_SUBJECT, SWEEP_STYLE),
            "negative_prompt": NEGATIVE,
            "num_inference_steps": steps,
            "width": width,
            "height": height,
            "seed": SEED,
        })
    return prompts


def build_resolution_sweep() -> list[dict]:
    prompts = []
    for width, height in RESOLUTION_VALUES:
        prompts.append({
            "prompt_id": f"resolution_sweep_{width}x{height}",
            "prompt_group": "resolution_sweep",
            "prompt": _prompt(SWEEP_SUBJECT, SWEEP_STYLE),
            "negative_prompt": NEGATIVE,
            "num_inference_steps": BASELINE_STEPS,
            "width": width,
            "height": height,
            "seed": SEED,
        })
    return prompts


def build_prompt_set() -> list[dict]:
    return build_baseline() + build_step_sweep() + build_resolution_sweep()


if __name__ == "__main__":
    out_path = Path(__file__).parent / "prompts.jsonl"
    prompt_set = build_prompt_set()
    with out_path.open("w") as f:
        for p in prompt_set:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(prompt_set)} prompts to {out_path}")
