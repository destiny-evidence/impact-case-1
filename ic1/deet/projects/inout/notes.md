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

## Scope specification (current design — supersedes "The four modes" above)

The four-mode design was consolidated to **three scopes** built as one shared **base prompt**
(`best balance`) plus small, localized **departures**. `max precision` was dropped (it collapsed
onto the others). All three live in `prompts/prompt_definitions.csv`; each is judged in its **own
LLM call** with **self-consistency voting**. Bundling all three prompts in one call lets the strict
`high precision` block anchor the others toward exclusion (measured ~0.5→0.86 recall swing on
`best balance` depending on prompt order), so scopes must be scored independently:
`method: llm_per_attribute`, `votes: N` (odd; early-stop majority) in the extraction config.

### Shared core (identical in all three scopes)

INCLUDE only if the record has **both**:
- a **CLIMATE COMPONENT** — a substantive (measured/modelled/analysed/used, not merely named)
  climate or weather factor, *or* a mitigation action, *or* an adaptation action. Fossil fuels
  count across their lifecycle (extraction…use); particulate matter, incl. wildfire/biomass smoke,
  counts as a climate-forcing agent.
- a **HEALTH COMPONENT** — a connection to human health/wellbeing (adverse *or* protective),
  via a direct human-health outcome *or* a recognised climate–health exposure pathway (air quality,
  extreme weather, food-for-human-consumption, heat, psychosocial, vectors, water-as-hazard/resource).

Shared validity floors (all scopes): the health outcome must be **in people**, not solely an animal
or in-vitro/cell model; ecological-object studies (e.g. diatom bioindicators) are not a health
connection.

### The three scopes — exact differences

All share the core above and differ **only** in the INCLUDE instruction (plus, for precision, two
definition tweaks). The scopes **nest**: high recall ⊇ best balance ⊇ high precision.

- **high recall** = base + (1) a mitigation *or* adaptation action counts on its own, with **no
  separate health component**; (2) include on any plausible connection, exclude only when none.
  - SHOULD contain: everything best-balance does, **plus** any mitigation/adaptation action even
    with no health link (emission-reduction tech, wind farms, DRR/policy).
  - SHOULD NOT: records with no plausible climate factor at all, or no plausible
    health/mitigation/adaptation link (pure clinical with no climate; mechanical/ecological studies).

- **best balance** (the base / reference) — requires **both** components; each may be explicit
  *or directly inferable*; lean include.
  - SHOULD contain: climate + health at a plausible/inferable level; mitigation/adaptation **only
    when they carry a health nexus**.
  - SHOULD NOT: mitigation/adaptation with no health link; pure emissions/energy/engineering;
    incidental climate mentions.

- **high precision** = base + (1) both components must be **explicitly stated**, not inferable —
  exclude when implied/uncertain; (2) fossil fuels count only via **"use"** (not
  extraction/production/refining), so occupational/industrial fossil-fuel cases drop out;
  (3) exposure pathways require the record to show **people actually exposed/affected**, not mere
  presence of the factor.
  - SHOULD contain: only records with an explicit climate factor **and** an explicit human-health
    outcome/exposure connected to it.
  - SHOULD NOT: anything inferred — on-mention PM/food/water; mitigation/adaptation without a stated
    health outcome; hazard/engineering without a stated human effect.

**The single scope-defining override** (`best balance` vs `high recall`): the guidance says "any
mitigation action is health-relevant," but the annotated corpus **majority-excludes**
mitigation/adaptation without a health nexus. `high recall` follows the guidance (includes them);
`best balance` follows the corpus (requires the nexus). That override is the main boundary.

### Maintenance

Edit `best balance` (the base), then re-sync the two departure blocks into `high recall` and
`high precision`. The shared core stays identical across all three.

### Measured operating points (dev 130, 5-vote per-attribute, 2026-09-01, Luna)

| scope | P | R |
|---|---|---|
| high recall | 0.46 | 0.93 |
| best balance | 0.67 | 0.71 (median draw ~0.76–0.78) |
| high precision | 0.80 | 0.29 |

`best balance` recall wobbles run-to-run because ~4 records (tides/shoreline flood-adaptation,
Sendai DRR) sit at ~50% — genuine coin-flips voting cannot pin, matching the human label
inconsistency. A **70-doc held-out validation degraded and was rejected** (2026-09-01); those docs
are now available to iterate on. Residual errors at every scope remain the label-floor twins (table
above); **QGIS-drainage** joins the flood-adaptation twin class (same as tides/shoreline), and
**radar obstacle-detection** is a genuine over-reach (incidental weather + vehicle-safety read as
climate + health).

### Validation FN analysis (2026-09-02, best balance)

Working the 4 `best balance` false negatives from the rejected 70-doc validation:

- **Water reserves** (*Future Terrestrial Water Reserves…*, 62822866) — **fixable and fixed.** The
  "water" pathway wording was too literal (only "water people use or drink"); scarcity/availability
  is inherently human-facing. Clarifying the pathway to "…contamination, water scarcity/availability,
  or water people use, drink, or depend on" flips it 0→5/5 INCLUDE (harness, Luna, N=5) with no
  collateral on guards (metallic-glass, obstetric hold EXCLUDE).
- **Free-flowing rivers** (*…global biodiversity targets*, 88c97b5e) — **out of scope for best
  balance and high precision; belongs in high recall only.** Framed as ecological integrity /
  biodiversity, not human water use — no health nexus in the abstract. The human INCLUDE is a
  label-floor case; only high recall (mitigation/adaptation without a health link) should capture it.
  Unmoved by the water-scarcity change (0/5), correctly.
- **Chinook salmon** (habitat/productivity, c98f4925) — wild-fishery habitat study, no explicit
  human food-security/consumption link; same label-floor character. Not chased (would need to widen
  the fisheries boundary, risking ecological-fish collateral).

Also fixed this session: **Sendai / DRR science-policy** FN (0e3bc1fb). The LLM granted the
climate/adaptation component but excluded on the **health** side ("health named only as broad
benefit"). Added a generic HEALTH-COMPONENT branch to best balance + high precision:
*"a climate mitigation or adaptation action whose stated aim includes protecting human lives, health,
health systems, or livelihoods **from climate-related harms**; the thing protected must be people or
health, not property/assets/economic or energy-supply continuity."* Flips Sendai 0→3–5/5 INCLUDE.
Guards held EXCLUDE: property-protection twin (Swiss structural insurance, 966f36ec), oil-trade-network
resilience (energy security), pure mitigation-tech twins (solar-carbothermic zinc, wind-farm, Regime
Interaction). NB deliberately generic — do **not** name Sendai in the prompt.

**The "from climate-related harms" qualifier is load-bearing.** Without it, the first-committed
version ("protecting human lives, health…") caused a **radar** collateral FP (202d1bc8, vehicle
obstacle-detection): the bare "protecting human lives" read vehicle-*safety* as health-protection,
flipping radar 1/5→5/5 INCLUDE (confirmed causal: with-bullet 5/5, without-bullet 1/5, in a dev run).
Adding "from climate-related harms" restored radar to 1/5 (collision ≠ climate harm) while keeping
Sendai 3/5 and Swiss 0/5. Lesson: scope protective-aim language to *climate* harms, or safety/security
framings leak in.

### High-precision scope analysis (2026-09-02) — settled, do not re-litigate

Dev run (200 docs, water_scarcity): high precision **TP=7, FP=2, FN=17** → P=0.78, R=0.29. P is
barely above best balance's 0.75 **because TPs are few, not because FPs are many** (only 2 FP on 9
predicted-positives — a noisy estimate).

- **The 17 FNs are not recoverable.** HP grants the climate component in every one and drops them
  on the health side. All are either validity-floor drops (health in **animals/plants** — dietary
  sulphur 3d63f91c, olive anthracnose 67ee0971) or **health-inferred** cases (water scarcity, floods/
  surges, black carbon & PM air quality, crop/food, peatland haze, youth-theatre psychosocial). That
  is precisely what HP's "explicitly stated, not inferable" bar is built to sacrifice. Recovering any
  would require loosening the health bar, which readmits the rest and collapses HP toward best balance.
  → **Low recall is by design, not fixable error.**
- **The 2 FPs resist clean removal.** Both (industrial gas-pipeline accident 05d82313; Georgia resort
  b4bd891c) also survive in best balance. Fossil-fuel clause reword attempts on the pipeline:
  - "use" → "combustion": pipeline **survives** (rupture ignition/thermal read as combustion) AND
    drops the petroleum-industry TP (216356f8, occupational). Net worse.
  - "use" → "intentional combustion": pipeline **dies (0/5)** but also drops the **air-pollution
    respiratory TP** (4e7da8ca) — ambient combustion emissions not read as "intentional." −2 TP / +1 FP. Worse.
  - "use" → "use as fuels": pipeline **survives (4/5)**, TPs kept — ≈ status quo, just noisier.
  The clause can't separate the pipeline from legitimate combustion-emission TPs. Industrial-accident-
  vs-occupational-exposure carve-out rejected as indefensible.

**Decision:** leave high precision's fossil-fuel clause as "use"; accept P≈0.78 / R≈0.29. HP is the
strict anchor working as intended. Its headline P is noise-dominated by 2 entangled FPs; chasing them
costs more (TPs, collateral) than the cosmetic gain is worth.

### Food pathway clarification (2026-09-02) — best balance + high recall ONLY, not high precision

Best-balance FNs olive-anthracnose (67ee0971), seed-priming-under-drought (e1052446), and salmon
(c98f4925) were dropped because the food bullet ended `-> nutrition, food security, foodborne disease`
— the model read those as **required human outcomes** and treated crop disease / fish productivity as
"an outcome in the plant/fish, not people" (over-extending the animal/in-vitro validity floor two
lines up). The listing of food crops/fisheries wasn't enough; the model wanted the record to make the
crop→human step.

**Fix:** reworded the food bullet so the food source itself is the exposure locus —
*"a climate or weather effect on, or an action protecting, the yield, quality, safety, or supply of a
food crop, livestock, dairy herd, or fishery engages this pathway (the affected food source is the
exposure; a food crop, food fish, or livestock is understood to be for people)."* Deliberately does
**not** say "no human outcome required" (that would soften high precision). Best balance: olive + seed
-priming → **5/5 INCLUDE** (salmon stays 1/5 — ecological framing, acceptable loss). Collateral guards
held **0/5 EXCLUDE**: plant-gametophyte/temperature (542f56a5), pollinator/passion-fruit (1310ced0),
non-climate plant-protection agronomy (ccb24fb6).

**Why NOT in high precision (scope divergence — intentional, not drift):** the reword was meant to be
scope-differentiated by the existing pathway preambles (BB "even if no clinical outcome named" vs HP
"people actually exposed/affected"). **This failed empirically** — in HP the bullet leaked past its
own preamble: olive/seed-priming went 3/5 INCLUDE and, decisively, the **polymer crop-agronomy EXCLUDE
twin (4cedc106) went 5/5 INCLUDE** — a solid FP in the one scope that can't afford it. So the food
reword is committed to **best balance (line ~124) + high recall (line ~59) only; high precision (line
~194) keeps the original "Food produced for human consumption…" bullet.** HP's food line now diverges
from the other two by design — the "identical pathway list across scopes" maintenance rule no longer
holds for the food bullet.

**Accepted trade (recall lean, β>1):** the crop-agronomy twins are irreducible — no prose separates
olive (want IN) from the polymer twin (labelled OUT). Best balance now includes both: +2 TP (olive,
seed-priming) for +1 known FP (polymer). Dietary-sulphur (3d63f91c) already IN via best balance;
its human-INCL is an animal-health label quirk HP still (correctly) drops.