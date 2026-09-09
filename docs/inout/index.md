---
title: Relevance screening
---

## Human annotation

After defining the system boundaries for inclusion in the Destiny Repository,
we annotated 5,000 documents from the ~12m results of our search query: [](annotation.md)

## Automated screening

To produce annotations for the remaining records, we explore and evaluate combinations of 
machine learning and LLMs.

- [Machine-learning](./ml.md)
- [LLM screening](./llm.md)
- [Combining ML + LLM](./combination.md)

## Balancing precision and recall

**Precision** measures the proportion of records included in the repository that were truly eligible. Low precision means that repository users see a high number of irrelevant records.

**Recall** measures the proportion of truly eligible records that are included in the repository. Low recall means that the repository misses a high number of eligible records that it should have included.

Balancing precision and recall is always a trade-off, as optimising recall generally means being more *inclusive* (and so including more irrelevant records), whereas optimising precision means being more *exclusive* (and so excluding more irrelevant records).

Different users of the repository will have different preferences for precision and recall. For example, researchers using the repository as a basis for systematic reviews need the repository to have high recall (at the cost of lower precision), while policymakers who do not have time to sift through irrrelevant or marginally relevant records will prefer a repository with high precision (at the cost of recall).

We therefore employ 3 classiciation modes, or operating points: 
- a high precision mode, 
- a high recall mode, 
- and a high recall mode.

Each is evaluated separately.