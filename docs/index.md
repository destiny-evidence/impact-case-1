---
title: DESTINY IC1 —  screening and taxonomy-annotation of climate and health studies
subtitle: Prompt-engineering a climate-and-health evidence pipeline
---

DESTINY builds a living evidence base at the intersection of **climate change** —
its impacts, mitigation, and adaptation — and **human health**. This report
documents how we use machine learning (ML) and large language models (LLMs) to two ends:

- **Relevance screening (in/out)** — deciding whether a bibliographic record
  belongs in the repository at all, at several precision/recall operating points.
  See [](./inout.md).
- **Taxonomy classification** — coding the included records against a controlled
  vocabulary of climate-and-health concepts. See [](./taxonomy.md).

For both, the LLM pipeline is build and evaluated with same evaluation pipeline (`deet`). 
Here we use the the same working method: treating prompt engineering as an empirical, 
measured activity, iterating
against a held-out gold standard, and recording every step. 
The [method chapter](./method.md) describes that shared machinery; the two task
chapters describe the iterative process for each task.
