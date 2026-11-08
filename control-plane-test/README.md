# control-plane-test

Checks the platform itself, not the models: the API answers, the
catalogue is readable, and jobs pass through the queue and the database
to a result. It uses the smallest models available, so it is fast and
downloads little. Run it first on a new host.

Model execution -- every model type, the OpenAI-compatible endpoints,
binary outputs -- is `platform-model-test`'s job.

## Models

- `qwen/qwen3-0.6b` -- instruct
- `huggingfacetb/smollm2-135m` -- instruct
- `sentence-transformers/all-minilm-l6-v2` -- text-embedding
- `openai/clip-vit-base-patch32` -- image-embedding
- `huggingfacetb/smolvlm-256m-instruct` -- img2txt

## Files

- `run.py` -- the checks; exits 0 when every check passes.
- `models.yaml` -- the models above.
