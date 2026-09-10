# benchmark-vid-gen

Submits a fixed prompt set to every txt2vid model in a running Marigold
deployment and records timing, VRAM, and power draw per clip.

Sized for the cost. Eighteen prompts across three models is 54 clips, and
a clip is minutes rather than seconds. Run the smoke test first.

## Setup

```bash
pip install requests pyyaml
```

## Run

```bash
marigold cache populate marigold-examples/benchmark-vid-gen
marigold deployment start marigold-examples/benchmark-vid-gen

python run_benchmark.py --base-url http://localhost:8000 \
  --user-id benchmark --max-prompts 3 --out results/smoke.csv
```

The smoke test is three clips per model. Check the timings before
running the full set, and multiply.

## Prompt groups

- `baseline` -- ten subjects, 30 steps, 49 frames at 8 fps, 832x480.
- `frame_sweep` -- 17 through 65 frames. The axis with no image
  equivalent, and the one most likely to exhaust VRAM: attention runs
  over the whole clip, so memory grows faster than frame count.
- `step_sweep` -- 10 through 50 steps.

Subjects all contain motion. A still scene gives a video model nothing to
do and makes frame count meaningless as a measurement.

## Options

Defaults differ from the image benchmark:

- `--workers 1` -- a single clip saturates the GPU. Concurrency here adds
  polling traffic and nothing else.
- `--wait-seconds 3600` -- a cold load plus a long clip exceeds any
  shorter deadline.
- `--poll-interval 15` -- polling a job that takes minutes every five
  seconds is noise.

## Clips

Binary output retrieval returns 501, so the CSV records the output
reference. Files are on the machine running the worker:

```
$MARIGOLD_CACHE_DIR/data/outputs/txt2vid/
```

Clips are tens of megabytes each. Check disk before a full run.
