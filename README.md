# marigold-examples

Example packages for [bayis-marigold](https://github.com/bayinfosys/marigold) --
Marigold's model-serving stack. Each directory here is a self-contained
Marigold package: a `models.yaml` declaring what it needs, a
`marigold.toml` declaring how to run it, and (where relevant) a script
that exercises it.

## Quick start

```bash
pip install bayis-marigold
git clone https://github.com/bayinfosys/marigold-examples
marigold deployment start marigold-examples/chat
```

## Examples

| Directory | Description | Full walkthrough |
|---|---|---|
| [`chat`](chat/) | Private chat via Open WebUI | [marigold.run/examples/chat.html](https://marigold.run/examples/chat.html) |
| [`simple-rag`](simple-rag/) | Chat + retrieval, with a built-in verification test | [marigold.run/examples/simple-rag.html](https://marigold.run/examples/simple-rag.html) |
| [`quick-platform-test`](quick-platform-test/) | Smallest example, fast first-run check | [marigold.run/examples/quick-platform-test.html](https://marigold.run/examples/quick-platform-test.html) |
| [`benchmark-llm`](benchmark-llm/) | Power and latency benchmarking across models | [marigold.run/examples/benchmark-llm.html](https://marigold.run/examples/benchmark-llm.html) |
| [`agents/langgraph`](agents/langgraph/) | LangGraph agent setup | *incomplete -- no `models.yaml` yet* |

See the [main repo](https://github.com/bayinfosys/marigold) for
installation, configuration, and CLI reference.
