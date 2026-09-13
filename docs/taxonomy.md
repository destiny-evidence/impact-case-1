---
title: Taxonomy classification
---

Once a record is included, we code it against the DESTINY controlled vocabulary —
a taxonomy of climate-and-health concepts. Each concept is a yes/no decision, and
a concept's prompt is built directly from its definition and scope note in the
taxonomy source (a TTL file).

## Prompt Iteration

```{figure} figures/taxonomy/iteration_timeline.svg
:name: taxonomy-timeline
:width: 100%

Prompt-engineering iterations for taxonomy classification. **Top two panels:**
micro- and macro-F1 over scoreable concepts, across every experiment in
chronological order. **Colour** = method (flat vs top-down cascade, plus keyword
and semantic comparators); **marker shape** = model. The shaded band on the left
is the pre-prompt-engineering baseline (pruned taxonomy). **Bottom panel:**
per-step prompt edits — words added (up) or removed (down) to the concept prompts
(teal) and the system prompt (purple).
```

{ref}`taxonomy-timeline` shows the gradual improvement in performance 
as edits were made to the taxonomy, as well as the variation in performance
between different models  and approaches. The 4 approaches tested were

- flat: the default deet mode where all prompts are sent to an llm in one call, along with the document
- top-down: instead of one call, deet classifies in stages that follow the concept hierarchy, with top level concpets first, followed by the children of only those concepts which were returned as relevant from the previous iteration.
- keyword: where deet returns concepts where any of the phrases contained in a concept's alt_labels field are contained in the document text
- semantic: where deet returns concepts where any of the phrases contained in a concept's alt_labels field have a semantic similarity (using a specified embedding model) to any of the document's sentences greater than a specified threshold.

## Cost vs performance trade-off


```{figure} figures/taxonomy/cost_performance.svg
:name: taxonomy-pareto
:width: 100%

Cost performance trade-off. Dotted line shows the pareto frontier.
```

{ref}`taxonomy-pareto` shows each considered model or approach to classification in terms of its cost and performance. We select the Luna model with the top-down approach to classification, as the vast increases in costs that moving to the larger sol model would not justify the very moderate increase in performance.


## Classifier performance by scheme



```{figure} figures/taxonomy/level_scores.svg
:name: taxonomy-level-scores
:width: 100%

F1 in schemes and sub-concepts. Each **grey bar** is a scheme's rolled-up F1 over all
its concepts; each **dot** is one internal node's F1 over its direct children
only, coloured by depth and sized by child count.
```

### Detailed performance by scheme, and sub-concepts


:::{include} tables/taxonomy_drilldown.md
:::
