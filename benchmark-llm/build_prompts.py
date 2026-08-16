"""
build_prompts.py -- prompt set for the Marigold model-eval benchmark.

Three groups, generated combinatorially (topics x phrasing templates)
rather than hand-typed one by one -- growing the dataset from here means
editing the TOPICS/templates lists, not writing more prompts.

  varying_context -- system prompt length increases (short, medium, long).
                      Tests whether cost scales with prompt size.

  structured       -- JSON-schema-constrained responses. Tests structured
                       output overhead and validity rate.

  long_form        -- explicit "answer at length" prompts, paired at run
                       time with a high max_tokens. Sustains GPU load long
                       enough for power/VRAM sampling to mean something.

The old fixed_prefix group (testing prefix/KV-cache reuse via a TTFT
delta) is not included: Marigold's /gen/instruct path doesn't stream,
so ttft_seconds is always None here -- that group would write rows with
no signal in the one column it exists to populate.
"""

import json
from pathlib import Path

TOPICS = [
    "prefix caching", "continuous batching", "quantisation formats",
    "KV cache eviction", "structured output overhead", "model cold-load time",
    "GPU memory fragmentation", "speculative decoding", "batch scheduling",
    "tokenizer overhead", "context window limits", "multi-GPU sharding",
    "power draw under sustained load", "throughput vs latency tradeoffs",
    "request queueing under concurrency", "cache-aware routing",
    "model quantisation error accumulation", "attention kernel selection",
    "cold-start latency", "memory-mapped weight loading",
]

VARYING_CONTEXT_SYSTEMS = {
    "short": "You are a helpful technical assistant. Be concise.",
    "medium": (
        "You are a technical assistant supporting engineers who operate "
        "private LLM inference infrastructure. You answer questions about "
        "serving frameworks, container orchestration, GPU memory, and "
        "deployment failure modes. Keep answers under four sentences, in "
        "plain prose, British English."
    ),
    "long": (
        "You are a technical assistant supporting engineers who operate "
        "private LLM inference infrastructure, including container "
        "orchestration, GPU-backed compute, and message-queue-driven job "
        "processing. The infrastructure serves heterogeneous model types "
        "-- large language models, embedding models, image generation, "
        "and text-to-speech -- each with different memory footprints, "
        "loading characteristics, and failure modes. You have deep "
        "familiarity with continuous batching, KV cache management, "
        "quantisation formats, and the operational differences between "
        "serving frameworks. Answer questions precisely, in plain prose, "
        "British English, no bullet points, no more than five sentences, "
        "and flag any uncertainty rather than guessing."
    ),
}

VARYING_CONTEXT_TEMPLATES = [
    "Explain how {topic} works in a modern LLM serving stack.",
    "What problem does {topic} solve, and what does it cost to implement?",
    "Why might two serving frameworks handle {topic} differently?",
    "What would go wrong if {topic} were implemented incorrectly?",
    "Describe a scenario where {topic} matters more than it usually does.",
]

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "category": {"type": "string", "enum": ["infrastructure", "model", "tooling", "cost"]},
        "confidence": {"type": "number"},
    },
    "required": ["summary", "category", "confidence"],
}

STRUCTURED_TEMPLATES = [
    "Classify and summarise: {topic} in a production inference deployment.",
    "Provide a JSON classification and one-sentence summary for: a system that relies heavily on {topic}.",
]

LONG_FORM_TEMPLATES = [
    "Write a detailed, multi-paragraph explanation of {topic} in modern LLM serving, covering the underlying mechanism and its practical trade-offs.",
    "Explain {topic} in depth: what it is, why it matters at scale, and what can go wrong when it's handled poorly.",
]


def build_varying_context() -> list[dict]:
    prompts, idx = [], 0
    for length_label, system_text in VARYING_CONTEXT_SYSTEMS.items():
        for topic in TOPICS:
            for template in VARYING_CONTEXT_TEMPLATES:
                prompts.append({
                    "prompt_id": f"varying_context_{length_label}_{idx:03d}",
                    "prompt_group": f"varying_context_{length_label}",
                    "system": system_text,
                    "user": template.format(topic=topic),
                    "json_schema": None,
                    "max_tokens": 256,
                })
                idx += 1
    return prompts


def build_structured() -> list[dict]:
    prompts, idx = [], 0
    for topic in TOPICS:
        for template in STRUCTURED_TEMPLATES:
            prompts.append({
                "prompt_id": f"structured_{idx:03d}",
                "prompt_group": "structured",
                "system": "Respond only with valid JSON matching the requested schema.",
                "user": template.format(topic=topic),
                "json_schema": STRUCTURED_SCHEMA,
                "max_tokens": 512,
            })
            idx += 1
    return prompts


def build_long_form() -> list[dict]:
    prompts, idx = [], 0
    for topic in TOPICS:
        for template in LONG_FORM_TEMPLATES:
            prompts.append({
                "prompt_id": f"long_form_{idx:03d}",
                "prompt_group": "long_form",
                "system": "Answer thoroughly and in depth. Plain prose, British English.",
                "user": template.format(topic=topic),
                "json_schema": None,
                "max_tokens": 4000,
            })
            idx += 1
    return prompts


def build_prompt_set() -> list[dict]:
    return build_varying_context() + build_structured() + build_long_form()


if __name__ == "__main__":
    out_path = Path(__file__).parent / "prompts.jsonl"
    prompt_set = build_prompt_set()
    with out_path.open("w") as f:
        for p in prompt_set:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(prompt_set)} prompts to {out_path}")
