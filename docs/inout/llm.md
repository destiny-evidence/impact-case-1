---
title: LLM classification
---

The first stage decides whether a record belongs in the repository at all. Rather
than eliciting a single graded score and sweeping a threshold, we define several
**operating points** as explicit prose scopes — each spelling out what the screen
purposely includes and excludes:

- **high recall** — any climate signal plus any health/mitigation/adaptation
  signal; err toward include.
- **best balance** — a substantive climate factor plus a plausibly
  health-relevant route; the headline operating point.
- **high precision** — both the climate element and the health connection must be
  explicitly stated; exclude when uncertain.

## Test results

On held-out test data, each operating point behaved as expected. 
Recall was highest in the high-recall setting, lower in the best balance, 
and lower again in high precision, with the reverse being true for precision.

High recall mode achieves an F2 score (which weights recall more strongly than precision)
of 0.80, with an F1 score of 0.72.

:::{include} ../tables/inout_test_metrics.md
:::


## Cost vs performance

```{figure} ../figures/inout/cost_performance.svg
:name: inout-timeline
:width: 100%

Cost performance trade-off
```


## Prompt Iteration

```{figure} ../figures/inout/iteration_timeline.svg
:name: inout-timeline
:width: 100%

Prompt-engineering iterations for the in/out screen. **Top two panels:**
precision and recall.
**Colour** = operating mode; **marker shape** = model. **Shaded bands** mark the
dev / validation / test cycles, labelled with the number of documents evaluated
(the set grows as rejected validation docs fold back into development). **Bottom
panel:** per-step edits to the scope prompts (teal) and the system prompt
(purple).
```
