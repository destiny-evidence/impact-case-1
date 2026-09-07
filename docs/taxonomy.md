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

Reading left to right: the baseline runs on the unedited (pruned) taxonomy sit
lowest. Performance climbs as each named intervention lands — clarifying the
population scope notes, the topic and intervention definitions, broadening the
technology/infrastructure and institutional concepts — with each climb sitting
directly above the prompt edit that produced it in the bottom panel.

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
