---
title: Combining ML + LLM
---

```{figure} ../figures/inout/comparison.svg
:name: inout-timeline
:width: 100%

```

- ML-only (high recall) and LLM-only score equally highly on recall, with
LLM-only performing better on precision.
LLM-only is best for F2.
- ML-only and ML->LLM (high recall) perform equally highly on F2,
but cascade wins on F1.
- Cascade has lower recall than ML-only, and than LLM-only. 
Presumably LLM and ML miss different records.
- Cascade is best overall for F1, and has better precision than LLM alone.
Implies that LLM has FPs that filtering removes.

:::{include} ../tables/inout_comparison.md
:::