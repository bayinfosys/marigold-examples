"""
build_prompts.py -- prompt set for the Marigold video generation benchmark.

Video generation cost scales with frames as well as steps and pixels,
and every axis is expensive: a single clip is minutes, not seconds. The
prompt set is therefore small and deliberately sparse. Ten subjects at
one setting, plus two short sweeps, is roughly 20 clips per model.

  baseline    -- fixed steps, frames and resolution, varied subject.

  frame_sweep -- one prompt, frame count varied. The axis that has no
                 equivalent in image generation, and the one most likely
                 to hit the VRAM ceiling, since attention is over the
                 whole clip.

  step_sweep  -- one prompt, step count varied. Fewer points than the
                 image benchmark: each one costs a full clip.

Subjects are chosen to contain motion. A still scene gives a video model
nothing to do and makes frame count meaningless as a measurement.
"""

import json
from pathlib import Path

SEED = 20260910

BASELINE_STEPS = 30
BASELINE_FRAMES = 49
BASELINE_FPS = 8
BASELINE_SIZE = (832, 480)

SUBJECTS = [
    "waves breaking against a harbour wall, spray in the wind",
    "a red fox running through long grass",
    "rain falling on a terraced street, water running along the kerb",
    "clouds moving quickly over a green hillside",
    "a steam locomotive pulling away from a platform",
    "leaves turning over in a gust beneath an oak tree",
    "a canal lock filling, water churning against the gates",
    "gulls circling above a fishing boat",
    "smoke rising from a chimney into a grey sky",
    "a flag snapping in strong wind on a headland",
]

STYLE = "cinematic, natural light, steady camera"

FRAME_VALUES = [17, 33, 49, 65]
STEP_VALUES = [10, 20, 30, 50]

SWEEP_SUBJECT = SUBJECTS[0]


def _prompt(subject: str) -> str:
    return f"{subject}. {STYLE}."


def build_baseline() -> list[dict]:
    width, height = BASELINE_SIZE
    return [
        {
            "prompt_id": f"baseline_{idx:03d}",
            "prompt_group": "baseline",
            "prompt": _prompt(subject),
            "negative_prompt": "",
            "num_inference_steps": BASELINE_STEPS,
            "num_frames": BASELINE_FRAMES,
            "fps": BASELINE_FPS,
            "width": width,
            "height": height,
            "seed": SEED,
        }
        for idx, subject in enumerate(SUBJECTS)
    ]


def build_frame_sweep() -> list[dict]:
    width, height = BASELINE_SIZE
    return [
        {
            "prompt_id": f"frame_sweep_{frames:03d}",
            "prompt_group": "frame_sweep",
            "prompt": _prompt(SWEEP_SUBJECT),
            "negative_prompt": "",
            "num_inference_steps": BASELINE_STEPS,
            "num_frames": frames,
            "fps": BASELINE_FPS,
            "width": width,
            "height": height,
            "seed": SEED,
        }
        for frames in FRAME_VALUES
    ]


def build_step_sweep() -> list[dict]:
    width, height = BASELINE_SIZE
    return [
        {
            "prompt_id": f"step_sweep_{steps:03d}",
            "prompt_group": "step_sweep",
            "prompt": _prompt(SWEEP_SUBJECT),
            "negative_prompt": "",
            "num_inference_steps": steps,
            "num_frames": BASELINE_FRAMES,
            "fps": BASELINE_FPS,
            "width": width,
            "height": height,
            "seed": SEED,
        }
        for steps in STEP_VALUES
    ]


def build_prompt_set() -> list[dict]:
    return build_baseline() + build_frame_sweep() + build_step_sweep()


if __name__ == "__main__":
    out_path = Path(__file__).parent / "prompts.jsonl"
    prompt_set = build_prompt_set()
    with out_path.open("w") as f:
        for p in prompt_set:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(prompt_set)} prompts to {out_path}")
