---
title: Combining ML + LLMs
---

## Summary

We pick the ML-filter -> LLM option, which reduces costs by 50% compared to LLM alone.

In order to achieve these cost savings, we sacrifice ~2 points of F2 (well within the margin of error), 
but achieve the highest F1 score (also well within the margin of error).

While the ML-filter -> LLM performs similarly to ML alone in terms of F2, it performs marginally better in terms of F1, due to significantly higher precision. 

An additional benefit of using the LLM for classification, is that we are able to implement separate scopes
that are each explainable.


```{figure} ../figures/inout/comparison.svg
:name: inout-timeline
:width: 100%
**Head-to-head on the held-out test set (648 documents, 83 relevant).** 
Point estimates and 95% highest-density intervals for precision, recall, F1 and F2,
computed from a Dirichlet–multinomial confusion-matrix posterior. 
**Colour** = LLM operating point (high recall / best balance / high precision); 
**marker shape** = system family: 
*LLM only* (the LLM decision on every document), 
*ML only* (the F-β–selected classifier, thresholded), 
*ML → LLM* (high-recall SVM filter gates the corpus, forwarded documents take the LLM decision — a cascade), and 
*ML or LLM* (include if the ML-only model *or* the LLM includes — an ensemble). 
ML-only carries no operating point and is shown in grey. 
Intervals overlap widely on precision (small positive count) but are tighter on recall; 
Cost is
not shown here — see {numref}`the comparison table <tbl-inout-comparison>`.
```

## Detailed results

### Prior

LLM-only would perform the best, an ML-filter optimised for recall could filter out obviously ineligible studies
LLM-only would not be likely to include, and so reduce costs.

Missed documents from ML-filter would likely be those that would be missed by LLM-only,
so that using ML-only with a filter would likely not degrade recall.

In other words, correlation between errors means combining methods is purely a cost-measure,
with minor impacts on performance.

### Findings

In fact, errors were not corrrelated, meaning that combining ML and LLMs has interesting implications for performance.

An option we had not considered, including all records where *either* the ML-only classifier
*or* the LLM classifier included a document, turned out to be the optimal pipeline
in terms of F2. 

This is because the False Negatives (FNs) from each method are not perfectly overlapping. 

Of the 20 FNs missed by either the ML-only or LLM-only methods, only 4 were missed
by both methods.

Errors are even more weakly correlated between the ML-filter and the LLM-only, where of the 16 FNs from either method, 
only 1 was shared between both.


::::{table} **System comparison on the held-out test set (648 documents).** Precision, recall, F1 and F2 as point estimate with 95% highest-density interval. *Sent to LLM* is the fraction of documents routed to the LLM (the cost driver); *Corpus cost* projects the LLM run's per-document spend to the ~6M-document corpus. Rows group by family, each combination family shown at all three operating points.
:name: tbl-inout-comparison

:::{include} ../tables/inout_comparison.md
:::
::::