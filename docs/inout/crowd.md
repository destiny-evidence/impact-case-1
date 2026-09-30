---
title: Crowd screening
---

## Summary

{ref}`inout-crowd` Shows the precision and recall of 1,000 crowd-generated labels.

```{figure} ../figures/inout/crowd_pr.svg
:name: inout-crowd
:width: 100%
**Crowd screening in precision–recall space.** The crowd (orange) with 95% highest-density intervals
from a Dirichlet–multinomial confusion-matrix posterior, over iso-F1 (grey) and iso-F2 (mauve,
recall-weighted) contours. Faded blue circles are the individual human coders and the faded markers
the automated systems (from {numref}`the head-to-head <inout-comparison>`), shown for context. The
crowd is scored over the 1,000 documents it saw — which span every split — whereas the systems are
scored on the 648-document held-out test set, so positions are indicative rather than strictly
like-for-like. The *test subset* point restricts the crowd to the held-out test documents for a
closer comparison.
```

:::{include} ../tables/inout_crowd.md
:::

## Notes

The crowd sample is not the held-out test set — its documents cut across the deet, train, validation
and test splits — so the headline row reflects the crowd's delivered performance on the sample it was
given rather than a split-clean benchmark. The *test subset* row restricts the crowd to the held-out
test documents so it can be read alongside the ML and LLM systems.
