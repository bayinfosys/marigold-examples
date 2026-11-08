# simple-rag

Verifies the full local RAG path: Marigold serving both chat and
embeddings, open-webui handling chunking, storage, and retrieval on top.
Confirms retrieval actually happens and discriminates correctly between
documents, rather than the model answering from its own knowledge.

## Models

`models.yaml` in this directory:

- `sentence-transformers/all-minilm-l6-v2` -- text-embedding, matches
  open-webui's `RAG_EMBEDDING_MODEL` with no further configuration.
- `qwen/qwen3-8b` -- instruct model.

## Running

```bash
marigold cache populate simple-rag
marigold application start simple-rag
```

Open `http://localhost:3000`.

## Test documents

Three short documents, each describing an invented entity with details
that cannot exist anywhere except in that file -- this is deliberate.
A real-world benchmark dataset (SQuAD, HotpotQA, BEIR) risks the model
having seen the source material during training, which would make a
correct answer ambiguous evidence: it wouldn't prove retrieval happened,
only that the model already knew it.

`doc1-bramblecroft.txt`:
```
Bramblecroft Systems was founded in 2019 by Elena Wraxford.
The company's internal support ticket system is called Nettleburn.
As of the last audit, Bramblecroft Systems had 342 employees
across offices in Hull and Ljubljana.
```

`doc2-corvidwatch.txt`:
```
Corvidwatch Ltd is a wildlife monitoring cooperative established in 2016.
Its founder, Tomasz Ferrenbeck, previously worked as a hydrologist.
Corvidwatch's flagship sensor platform is named Rookwire, and the
organisation currently operates 58 monitoring stations, concentrated
along the Danube tributary network.
```

`doc3-quillmere.txt`:
```
Quillmere Analytics is a data consultancy founded in 2021 by
Adaeze Obiako. The firm's proprietary forecasting model is called
Driftlantern. Quillmere reported revenue of 2.7 million euros in
its most recent financial year, with clients concentrated in the
insurance sector.
```

Upload all three together, so each test also confirms discrimination
between candidates, not just retrieval from a single uploaded file.

## Questions and expected results

| Question | Expected |
|---|---|
| What is Bramblecroft Systems' ticket system called, and how many employees does it have? | Nettleburn, 342 -- sourced to `doc1-bramblecroft.txt` only |
| Who founded Corvidwatch, and what's their sensor platform called? | Tomasz Ferrenbeck, Rookwire -- sourced to `doc2-corvidwatch.txt` only |
| What was Quillmere Analytics' revenue, and what's their forecasting model called? | 2.7 million euros, Driftlantern -- sourced to `doc3-quillmere.txt` only |
| What is Bramblecroft Systems' primary competitor? | Not stated in any document -- correct answer is "I don't know." This is the negative control: it checks whether the model fabricates an answer when nothing retrieved supports one, rather than checking retrieval itself. |

## What this confirms, and what it doesn't

Confirms: Marigold's embedding endpoint and open-webui's retrieval
pipeline are wired correctly, and retrieval discriminates between
multiple candidate documents rather than defaulting to whichever was
uploaded most recently.

Doesn't confirm: chunking quality or retrieval behaviour at realistic
document lengths or counts. Treat this as a plumbing check, not a
quality benchmark.

## Architecture note

Marigold has no vector store of its own. Retrieval, chunking, and
vector storage here are all open-webui's own ChromaDB instance,
created automatically under the shared model cache directory. Marigold
supplies embedding vectors on request and nothing else in this path.

