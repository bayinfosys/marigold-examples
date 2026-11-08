# benchmark-img-gen

Submits a fixed prompt set to every txt2img model in a running Marigold
deployment and records timing, VRAM, and power draw per image. Safe to
re-run -- each request's ID is a hash of its contents, and every field
in the body is fixed by the prompt set, so resubmitting already-completed
work is a cheap cache-status check rather than a re-run.

## Setup

```bash
pip install requests pyyaml
```

## Run

Populate the cache first. The full catalogue is roughly 96 GB and a cold
population is measured in hours, not minutes:

```bash
marigold cache populate benchmark-image-gen
marigold application start benchmark-image-gen
```

Then:

```bash
python run_benchmark.py \
  --base-url http://localhost:8000 \
  --user-id benchmark \
  --out results/run1.csv
```

`--user-id` is required and can be any string -- Marigold doesn't
require authentication, it just tags requests by this value.

Start with a smoke test before committing to the full set:

```bash
python run_benchmark.py --base-url http://localhost:8000 \
  --user-id benchmark --max-prompts 6 --out results/smoke.csv
```

## Prompt groups

Image generation cost is dominated by step count and pixel count. Each
group varies one axis.

- `baseline` -- 40 prompts, 25 steps, 1024x1024. Ten subjects across
  four styles. The cell everything else is compared against.
- `step_sweep` -- one prompt at 4, 8, 16, 25 and 40 steps. The linear
  term.
- `resolution_sweep` -- one prompt from 512x512 to 1536x1536. The
  quadratic term, and where the VRAM ceiling shows up.

Distilled models are in the step sweep deliberately. FLUX.1-schnell is
trained for four steps and z-image-turbo for a short schedule, so the
quality curve is flat above their intended count while the cost curve is
real. Read those rows as cost measurements.

Every prompt carries a fixed seed, so a re-run reproduces the same image.

## Options

- `--max-prompts N` -- stratified sample across the three groups.
- `--workers N` -- concurrent requests, default 4. The worker holds one
  model at a time, so requests queue regardless; this controls polling
  concurrency rather than parallel generation.
- `--wait-seconds N` -- how long to wait for a single job before writing
  it as still queued, default 900. A cold model load on a large DiT is
  minutes before the first image.
- `--models NAME ...` -- benchmark specific models instead of everything
  in `models.yaml`.
- `--config path.json` -- `base_url`, `api_key`, `user_id` from a file.

Requests are grouped by model rather than interleaved, since the worker
switches models on queue depth and interleaving would cost a full reload
between every request.

## Images

Binary output retrieval returns 501, so the CSV records the output
reference -- path and mimetype -- rather than the bytes. The files are on
the machine running the worker:

```
$MARIGOLD_CACHE_DIR/data/outputs/txt2img/
```

## Regenerating the prompt set

`prompts.jsonl` is already built and committed. Only needed if you're
changing the prompt set itself:

```bash
python build_prompts.py
```

Prompts are generated combinatorially from the SUBJECTS and STYLES lists
in `build_prompts.py`, not hand-written.
