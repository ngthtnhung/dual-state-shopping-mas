# Phase 2: Dual-conditioned laptop recommendation pipeline

Run commands from the workspace root. Phase 2 operates on the Phase 1 offline
catalog and filtered reviews; it does not call an external LLM or network API.

```powershell
python -m scripts.phase2 "laptop at least 16 GB RAM under $1000"
```

By default, the command reads `dataset/processed/amazon-laptops/laptops_catalog.jsonl`,
uses `dataset/processed/amazon-laptops/laptops_reviews.jsonl.gz` for review
evidence, and loads the SetFit model from `models/setfit-absa-laptops`. Use
`--catalog`, `--reviews`, `--model-dir`, `--top-k`, and
`--review-scan-limit` to override those paths and bounds. `--disable-review-agent`
keeps execution limited to metadata evidence.

## Components

1. `QueryAnalyzer` extracts explicit RAM, storage, display-size, USD price,
   battery-runtime, and weight constraints, normalizing supported units; it
   also records attribute targets, vague terms, and a deterministic
   per-attribute uncertainty heuristic.
2. `CatalogProfiler` streams the laptop metadata, filters hard constraints,
   ranks lexical matches with rating as a small tie-breaker, extracts
   source-linked specs, then reports candidate metadata completeness and
   normalized value entropy.
3. `AdaptiveOrchestrator` applies two gates. It asks a clarification only when
   query uncertainty, metadata coverage, and candidate-value diversity all meet
   configured thresholds. Otherwise, it early-exits for complete queries or
   activates specialist evidence paths for missing/experiential evidence.
4. `SpecDescriptionAgent` extracts cited values from metadata fields.
   `ReviewEvidenceAgent` scans a bounded prefix of the filtered review stream
   and calls the Phase 1 `ReviewABSAAgent`; each output retains the original
   `parent_asin`/variant `asin` and sentence offsets.
5. `ConflictEvaluator` flags negative review majorities against listed specs
   (`EXPERIENTIAL_DRIFT`) and opposite majority sentiment across variants
   (`VARIANT_CONFLICT`).
6. `DecisionSynthesizer` returns `ASK_CLARIFICATION`, `RECOMMEND`,
   `ABSTAIN_DISCLOSE`, or an explicit `ACTIVATE_EVIDENCE_AGENTS` next action,
   with metadata/review citations for recommendations.
7. `RunTrace` records the query/catalog state, route, agent calls, scan count,
   evidence, conflicts, and final decision as JSON.

## Heuristic and evaluation limits

The uncertainty scores, parser, lexical retrieval, regex spec extraction, and
review scan cap are deterministic heuristics. The default route thresholds
are calibrated on the fixed VACOS sample described below:
`U_Q >= 0.5`, candidate coverage `>= 0.5`, and normalized entropy `>= 0.3`.
These values remain provisional because vague-attribute matches are heuristic
proxy labels rather than human judgments.
Query parsing targets English, and numeric filters only apply to the currently
implemented units/attributes. Unknown values do not satisfy hard constraints.
The review path scans at most 50,000 review records per run by default and
collects at most six reviews per candidate; review order can bias this sample.
Sentiment is experiential evidence, not proof that a manufacturer specification
is incorrect. Validate and calibrate these choices on the VACOS scenarios and
manually judged evidence before reporting system-level performance.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```

## Threshold calibration and four-strata evaluation

Run the deterministic, balanced VACOS calibration/evaluation loop:

```powershell
python -m scripts.phase2.evaluate --sample-size 160 --seed 42
```

The script selects equal numbers of VACOS queries with and without a vague
annotation linked to a core attribute, restricted to queries where the current
analyzer recognizes at least one of Display, RAM, Battery, or
Weight/Portability. It makes a second balanced 50/50 calibration/held-out
evaluation split. It builds one
temporary FTS5 index over the frozen laptop catalog, profiles the selected
queries once, sweeps uncertainty `{0.3, 0.5, 0.7}`, coverage
`{0.25, 0.5, 0.75}`, and entropy `{0.1, 0.2, 0.3}`, then selects the threshold tuple with the best precision-weighted F0.5 score
(ties prefer lower Probe-Waste Rate, higher precision, then conservative
thresholds).

The script produces:

- `reports/phase2-evaluation/phase2_evaluation.json`: calibration grid,
  chosen thresholds, four-strata metrics, latency, agent dispatches, and
  per-query traces.
- `reports/phase2-evaluation/phase2_evaluation_cases.csv`: spreadsheet-ready
  per-query results.

Default mode measures **routing only** with a counting stub: it does not scan
reviews or run SetFit. It still measures catalog-profile and routing latency,
planned review-agent calls, and reports external LLM tokens as zero. To run
real SetFit review inference on a small sample (maximum 32 queries), pass
`--with-review-model`; `--review-scan-limit` bounds each review scan and exact
SetFit tokenizer input-token counts are recorded.

The held-out evaluation uses the natural catalog state for S1/S2, then makes
paired counterfactual copies of S1/S2 queries: S3 masks requested supported
attribute metadata for a clear query, and S4 masks it for an ambiguous query. This
guarantees coverage of the missing-evidence routes, but S3/S4 rows are
controlled stress tests, not additional independent user queries or evidence
found naturally in the catalog.

VACOS vague annotations do not directly identify their target attribute. The
script links each annotated character span to recognized core-attribute
mentions within a 24-character window, then uses that as a proxy label. The
clarification gold label additionally requires nonzero metadata coverage and
entropy for that attribute. S1--S4 membership is operationally derived from
those annotations, measured catalog state, and the selected thresholds, not
an independently annotated outcome. VACOS does not provide
gold recommendation, conflict, token-cost, or latency labels. Treat these
numbers as routing calibration results, not final comparative-system evidence.
