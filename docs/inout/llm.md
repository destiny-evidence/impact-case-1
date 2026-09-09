---
title: LLM classification
---

Observing the different interpretations of the annotation guidance visible in the annotated data,
and recognising the different preferences for precision and recall in repository users, we encode
our three precision/recall operating points explicitly in separate prompts for each mode (see [](llm-prompts)).

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

We used the dev-val-test splitting strategy to separate data used to evaluate an automated
pipeline from data used to develop and automated pipeline.

In this strategy, prompts and model configuration are iterated on using a set of development documents. 
Once the developer believes that further iteration will not be able to improve prompts,
the set of prompts is evaluated against a validation set, which measures the out-of-sample
performance of the developed prompts.

After observing validation statistics on the validation set, the developer must decide to
either:
- lock in that set of prompts and model configuration, and commit to a final evaluation on remaining documents which become the held-out test set, or
- fold the validation documents back into the development set, and continue iterating

The strategy is implemented in [deet](https://destiny-evidence.github.io/data-extraction-evaluation-toolkit/stable/).

{ref}`inout-timeline` shows the progress of prompt iteration and evaluation for the 
inclusion task. 
The first four data points show how edits to the system prompt and the best balance prompt 
contributed to an increase in precision and recall for the first set of 65 development documents.

The sharp drop precision in the first validation set shows that the first iteration cycle overfit the prompts
to the development data. 

We underwent two additional development-validation cycles, including a drastic shortening of the prompts visible 
in the large negative bar in the bottom panel of the figure (which shows changes to prompts), before arriving at 
the final scores as evaluated on the held-out test set.



## Cost vs performance

There was substantial variation in the cost of annotating 6m documents between models
of different sizes. 
We switched models periodically as part of prompt development and iteration, before
repeating the final validation evaluation with a range of different models, in order to
arrive at a fair comparison of how models performed, and how much they would cost.

```{figure} ../figures/inout/cost_performance.svg
:name: inout-pareto
:width: 100%

Cost performance trade-off. Dotted line shows the pareto frontier.
```

{ref}`inout-pareto` shows that more expensive models do not necessarily perform
better than cheaper models. 
In fact, although terra (the mid-range GPT5.6 model) performed slightly better
than luna (the smallest and cheapest GPT5.6 model), the perfomance increase
was minor, despite implying an 8x cost increase.

These two models formed the pareto frontier in the trade-off between cheapest and 
best, with luna (the model we selected for final evaluation and deployment) 
performing relatively well per dollar spent.

(llm-prompts)=
## Prompts

:::{include} ../tables/inout_prompts.md
:::
