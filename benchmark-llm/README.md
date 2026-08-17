# benchmark-llm

Submits a fixed prompt set to every instruct model in a running
Marigold deployment and records timing, token counts, VRAM, and power
draw per response. Safe to re-run -- each request's ID is a hash of its
contents, so resubmitting already-completed work is just a cheap
cache-status check, not a re-run.

Used to build the analysis in
[AI-Wales/build-llm-power-sampling](https://github.com/AI-Wales/build-llm-power-sampling).

## Setup

```bash
pip install requests
```

## Run

Start the deployment with the models you want to benchmark declared in
`models.yaml`:

```bash
marigold deployment start marigold-examples/benchmark-llm
```

Run the benchmark against it:

```bash
python run_benchmark.py --base-url http://localhost:8000 --user-id benchmark --out results/run1.csv
```

`--user-id` is required and can be any string -- Marigold doesn't
require authentication, it just tags requests by this value.

Re-run the same command later to pick up anything that finished since
-- results are appended fresh each run, nothing is skipped by
tracking state.

## Options

- `--max-prompts N` -- a stratified sample of N prompts spread across
  prompt groups, for a quick smoke test rather than the full set.
- `--workers N` -- concurrent requests (default 16).
- `--config path.json` -- a JSON file with `base_url`, `api_key`,
  `user_id`, as an alternative to passing them as flags.
- `--temperature` -- default 1.0.

Note: `--max-tokens` is accepted but currently has no effect -- each
prompt in `prompts.jsonl` carries its own `max_tokens`, which always
takes precedence.

## Regenerating the prompt set

`prompts.jsonl` is already built and committed. Only needed if you're
changing the prompt set itself:

```bash
python build_prompts.py
```

Three prompt groups: `varying_context` (system prompt length, short
through long), `structured` (JSON-schema-constrained output),
`long_form` (long generations, for sustained power/VRAM sampling).
Edit the topic and template lists in `build_prompts.py` to grow the
set -- prompts are generated combinatorially, not hand-written.

[Full walkthrough](https://marigold.run/examples/benchmark-llm.html)
