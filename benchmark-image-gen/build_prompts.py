"""
build_prompts.py -- parameter sets for the image generation benchmark.

Each line of prompts.jsonl is:

    {"label": "steps_004_r0", "request": {...}}

`request` is sent as the POST body verbatim, with `model` added by the
runner. `label` is recorded in the CSV and never sent.

Repeats are expressed here rather than in the runner. message_id is a
hash of the request body, so two identical bodies are the same job and
the second returns the first from cache. Each repeat carries a distinct
`nonce` for that reason -- and so does each sweep, because a cell in the
step sweep and a cell in the guidance sweep can otherwise resolve to the
same body. In the A10G run, 20 of 200 rows were cache hits from exactly
that collision and measured nothing.

Three axes, one parameter each.

  steps     the linear term. Measured slope on an A10G ranged from
            0.157 s/step (Sana 1.6B) to 2.534 (z-image), with a fixed
            overhead of 0.62-0.71 s on every model -- the VAE decode.

  guidance  binary rather than proportional. Classifier-free guidance
            runs two forward passes per step on a doubled batch, so
            cost roughly doubles when it engages and is flat either
            side. Where it engages varies: most models switch above
            1.0, z-image at 1.0, and FLUX never, being
            guidance-distilled.

  prompt    tokens of conditioning text. An open question rather than a
            known result: CLIP text encoders truncate at 77 tokens, so
            SD-family models should be flat across this axis, while T5
            and Gemma encoders accept far more and may not be. If the
            two groups diverge, that is the finding.

Prompt text is otherwise held constant. The same model at the same
settings executes the same graph with the same tensor shapes whatever
the words are, so varying content adds rows without adding information.
Content is deliberately plain: no style tokens, no photographic or
artistic vocabulary, nothing that a fine-tune would be advantaged or
disadvantaged by.

Resolution is absent because Txt2ImgRequest carries no width or height
and pydantic drops unknown fields, so a resolution sweep would vary
nothing. Adding those fields to the request and the handler is on the
backlog.
"""

import json
from pathlib import Path

SEED = 20260910

# One warm-up plus two measured. Across five repeats on the A10G, the
# median coefficient of variation was 0.02% and the first three gave the
# same mean as all five; the only real variation was the first request
# after a model load, which ran up to 7x slower. So the warm-up is worth
# having and the fourth and fifth repeats are not.
WARMUP_REPEATS = 1
MEASURED_REPEATS = 2

BASELINE_STEPS = 25
BASELINE_GUIDANCE = 7.5

STEP_VALUES = [1, 4, 8, 16, 25, 40, 100]
GUIDANCE_VALUES = [0.0, 1.0, 3.5, 7.5, 15.0]

# Plain declarative sentences about a scene. No style vocabulary, so no
# model in the set is closer to or further from its training
# distribution than any other. Lengths are approximately 8, 30, 90 and
# 220 CLIP tokens: the second sits under the 77-token CLIP limit, the
# third above it, the fourth well above.
PROMPT_FRAGMENTS = [
    "A stone castle on a hill.",

    "The castle has four square towers and a gatehouse facing south. "
    "The hill below it is covered in grass.",

    "A track runs from the gatehouse down the slope and turns west at "
    "the foot of the hill. There is a stand of trees beside the track "
    "and a low wall along its northern edge. The sky above is "
    "overcast.",

    "Beyond the wall the ground falls away into a shallow valley with "
    "a river running through it. A wooden bridge crosses the river at "
    "the narrowest point. On the far bank there are three stone "
    "cottages with slate roofs and a barn with its doors open. Sheep "
    "graze in the field behind the barn. A cart track leads from the "
    "bridge toward a line of hedgerows in the distance, and past them "
    "the land rises again toward a ridge topped with more trees.",
]

PROMPT_LENGTHS = {
    "008": 1,
    "030": 2,
    "090": 3,
    "220": 4,
}

BASELINE_PROMPT_KEY = "030"


def _prompt(fragments: int) -> str:
    return " ".join(PROMPT_FRAGMENTS[:fragments])


BASELINE_PROMPT = _prompt(PROMPT_LENGTHS[BASELINE_PROMPT_KEY])


def _cell(label: str, **overrides) -> dict:
    """One parameter set.

    The label is part of the nonce, so two cells that would otherwise
    produce identical bodies -- the 25-step cell and the 7.5-guidance
    cell, for instance -- stay distinct jobs rather than one job and one
    cache hit.
    """
    request = {
        "prompt": BASELINE_PROMPT,
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

    cells += [
        _cell(f"prompt_{key}", prompt=_prompt(fragments))
        for key, fragments in PROMPT_LENGTHS.items()
    ]

    # negative_prompt is passed to the pipeline unconditionally by
    # Txt2ImgModel._run, and several DiT pipelines do not accept the
    # kwarg. One cell, so a rejection shows up as a single failed row
    # rather than as a whole model missing.
    cells.append(
        _cell("negative_prompt", negative_prompt="blurry, low contrast")
    )

    return cells


def build_prompt_set() -> list[dict]:
    """Warm-up repeats first, then measured ones.

    The runner sends cells in file order per model, so the warm-up
    repeat of every cell runs before any measured repeat. That puts the
    model-load cost in rows labelled w0, which analysis excludes, rather
    than contaminating r0.
    """
    out = []
    cells = build_cells()

    for repeat in range(WARMUP_REPEATS):
        for cell in cells:
            request = dict(cell["request"], nonce=f"{cell['label']}_w{repeat}")
            out.append({
                "label": f"{cell['label']}_w{repeat}",
                "request": request,
            })

    for repeat in range(MEASURED_REPEATS):
        for cell in cells:
            request = dict(cell["request"], nonce=f"{cell['label']}_r{repeat}")
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

    cells = len(build_cells())
    print(f"wrote {len(prompt_set)} parameter sets to {out_path}")
    print(f"  {cells} cells x ({WARMUP_REPEATS} warm-up + {MEASURED_REPEATS} measured)")
