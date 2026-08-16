# marigold-examples

Example packages for [bayis-marigold](https://github.com/bayinfosys/marigold) --
Marigold's model-serving stack. Each directory is a self-contained
package: a `models.yaml` declaring what models it needs, a
`marigold.toml` declaring how to run it, and (where relevant) a script
that exercises it.

Install, configuration, and usage are documented in the main repo:
**[bayinfosys/marigold](https://github.com/bayinfosys/marigold)**.

```bash
git clone https://github.com/bayinfosys/marigold-examples
marigold deployment start marigold-examples/<directory>
```

## Examples

| Directory | Description |
|---|---|
| `quick-platform-test` | Smallest example. Exercises `/models`, `/embed/text`, `/embed/image`, `/gen/instruct`, `/gen/img2txt` against small models. Good first run. |
| `simple-rag` | Chat + text embedding with open-webui, set up for retrieval-augmented chat. Own `README.md` with test documents and expected Q&A pairs. |
| `chat` | Chat via open-webui. |
| `benchmark-llm` | Batch prompt benchmarking against a model set. |
| `agents/langgraph` | LangGraph-based agent setup. Incomplete. |
