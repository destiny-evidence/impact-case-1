# In/out screening — prompt modes, system boundaries, and annotation noise

Notes on how the DESTINY IC1 relevance (include/exclude) screen is being built with the
`deet` extraction pipeline.

## Approach: precision/recall as explicit prose boundaries, not a threshold

Rather than eliciting a graded score and sweeping a decision threshold, each operating
point is a **separate prose prompt** in `prompts/prompt_definitions.csv`. 

- Each prompt **explicitly encodes what the screen purposely includes and excludes** at
  that operating point.
- The modes correspond to genuinely different scope decisions (see below).
- Building them this way surfaced the **irreducible inconsistency in the gold labels**
  (next section).

Initial prompts iterated with Model: `gpt-5.6-sol` (Azure), `temperature: 1` 
(a reasoning model — temperature is not
adjustable, which matters for run-to-run variance; see Practical notes).

Gold: single human reference standard, ~1,000 records, ~12.5% prevalence. The three
`include - *` columns in the export are identical copies of one human decision (a fourth,
`include - max precision`, was duplicated so the fourth prompt can be scored).

## The four modes and the system boundary each encodes

From most inclusive to most conservative. All require BOTH (1) a climate-related element
and (2) a health/mitigation/adaptation element; they differ in how each is read.

- **high recall** — the *official annotation guidance applied literally, plus
  include-when-uncertain*. Any mitigation action counts (per the guidance's "any mitigation
  action is health-relevant"); thematic/policy climate mentions count; particulate matter
  qualifies on mention; exposure pathways count on mention; all-hazard disaster-risk-reduction
  counts as adaptation. Boundary: *any climate signal + any health/mitigation/adaptation
  signal; err toward include.* Recall-stage screen with a downstream filter assumed.

- **best balance** — the *a compromise between the literal guidance and the guidance as it was applied*. 
  Substantive climate factor + a
  health-relevant route: a direct or indirect (CCHE exposure-pathway) human-health outcome,
  a **health-relevant** mitigation action, or an adaptation action protecting human health,
  health systems, or human-important ecosystems. Exposure pathways must be **human-facing**
  (air is universal via breathing; water must be water people use, not ecological
  bioindication). Boundary: *substantive climate + a plausibly health-relevant route.*

- **high precision** — best balance **+ explicit-only + exclude-when-uncertain**. Both the
  climate element and the health-relevant connection must be explicitly stated, not inferred;
  air-quality qualifies only with a stated human-health outcome. Boundary: *explicitly stated
  climate + health-relevant connection.*

- **max precision** — include **only** records with an explicit, stated human-health outcome
  or risk from a climate factor/agent/fossil fuel (or an action explicitly stated to protect
  human health), plus a targeted exclusion of crop-yield agronomy. Boundary: *an explicit
  climate → human-health causal claim.* Nearly every include is correct; most includes are
  sacrificed.

## Operating points (dev set, 130 docs — single-run, treat as noisy)

| Mode | Precision | Recall |
|---|---|---|
| high recall | ~0.39 | ~1.00 |
| best balance | ~0.77 | ~0.71 |
| high precision | ~0.80–0.83 | ~0.57–0.71 |
| max precision | ~1.00 | ~0.50–0.57 |

These are dev-set (tuned-on) numbers and vary run to run (temperature 1). `best balance`
has been the most stable; `high precision` the most volatile (its exclude-when-uncertain
rule flips borderline records between runs). The apparent Pareto-dominance of `high
precision` over `best balance` seen in one run did **not** survive resampling — the two
overlap within noise. Final numbers must come from held-out test, not dev.

## Irreducible label noise: documents that make Pareto improvement impossible

The gold labels are internally inconsistent, for a known reason: the annotation guidance was
revised mid-project (the "any mitigation action is health-relevant" rule was added late,
after the opposite had been discussed), not all coders applied the latest version, and the
underlying search query targeted health pathways. So the actual labels diverge both from the
official guidance and from each other.

Concretely, several **near-identical pairs are labelled oppositely**. For each such pair,
no prose rule can include one without including the other, so beyond a point you cannot
raise precision without losing a true positive (or raise recall without gaining a false one).
These are the "twins" that cap each operating point:

| Category | INCLUDE (human) | EXCLUDE (human) | Why they can't be separated |
|---|---|---|---|
| Mitigation tech, no health | 2881444016 *Carbon Footprint of Maritime Fender Systems* | 852650 *Solar Carbothermic Zinc*; 1015 *Emission-Free Ferry*; 323408018 *Catalytic CFC removal*; 85383 *biomass-H2 with CO2 capture*; 86824372 *Regime Interaction and Climate Change* (aviation/maritime emission regulation) | All are emission-reduction technology/policy with no human-health nexus. |
| Crop / food agronomy | 754826 *Seaweed biostimulant reducing cherry cracking* | 7933919 *Polymer seed protection for grain crops* | Both protect crop yield from weather stress; neither states a human-health outcome. |
| Particulate matter / air | 93858 *Air pollution & respiratory health*; 347011 *PM2.5 bibliometric* | 506594 *PM source apportionment near a phosphorus plant* | All concern PM/air quality; the guidance treats PM as auto-qualifying, but the human labels split. |
| Water quality | 927648283 *Bloom-event environmental monitoring* | 4136132 *Diatoms as water-quality indicators (Karoo drought)* | Both are water-quality studies; separating them needs a human-water-use distinction the abstracts don't consistently state. |
| Thematic / societal climate | 8876 *Youth theatre and the climate crisis* | 3627 *Environmental justice / governance (Fiji)* | Both couple climate change to societal/policy response with no direct health outcome. |
| Thematic climate policy | 51599178 *Sendai DRR / science-policy* (INCLUDE) | (many thematic-policy records EXCLUDE) | Climate appears only as a co-named agreement; inclusion depends on a generous reading. |

Practical consequence: at any operating point, some residual errors are **label
inconsistencies, not model failures**. They set a precision/recall floor that is worth
*quantifying* but not worth *chasing* with more prompt tuning. 

## Practical notes

- **Temperature 1 → run-to-run variance.** Sol is a reasoning model; temperature can't be
  lowered. Borderline records flip between runs, most for `high precision`. If a
  precision-leaning mode is deployed, use **self-consistency** (N samples, majority vote per
  record) to stabilise it. Always report operating points as mean ± sd over several runs, and
  don't read a difference smaller than the noise band as real.

## Development process

Initial prompt development had not resulted in prompts that reach different operating
points on the precision / recall tradeoff. Further development reduced to one best balance prompt, and tried to find pareto improvements by identifying false positives 
and false negatives, and clarifying the base prompt where either could be eliminated without introducing more of the opposite. 

This process surfaced the labelling inconsistencies. Once pareto improvements were
no longer possible, we developed variations of prompts that would maximise precision
and recall respectively, recognising that the changes in system boundaries would reduce
precision when recall was maximised and vice versa. Claude code was used to identify FPs and FNs from deet output, and to suggest changes to prompts that could mitigate these according to the given target.