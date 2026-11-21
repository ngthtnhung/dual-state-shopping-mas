# Phase 4: annotation and ablation preparation

Export the Phase 3 responses and frozen case metadata into an Excel workbook:

```powershell
python -m scripts.phase4.prepare_annotation_workbook
```

By default this reads `reports/phase3/phase3_baseline_results.json` and writes
`reports/phase4/phase4_annotation.xlsx`. The workbook contains one row per
system/case response, the query and catalog-state fields, available decision
evidence, binary/manual annotation columns, metric summary formulas, a rubric,
and comparisons to the existing B4/B3 baselines.

Install the workbook dependency into the active interpreter if needed:

```powershell
python -m pip install -r requirements.txt
```

The human labels are intentionally left blank. The summary sheet calculates
CSR, recommendation coverage, binary unsupported-claim rate, EGP from claim
counts, probe-waste rate, disambiguation success, abstention quality, and
conflict-disclosure quality after labels are entered and the workbook is
recalculated in Excel or LibreOffice. Use blank for not-applicable labels as
described in the rubric.

The current data supports comparison with B4 as a query-only proxy for disabling
catalog awareness, and B3 as a fixed-pipeline proxy for disabling early exit.
These are baseline comparisons, not exact one-component ablations. The current
S3/S4 cases mask catalog metadata; they do not represent curated
manufacturer-versus-review contradictions. Therefore, the effect of disabling
the Review Agent and Conflict Evaluator cannot be established from this run.
That ablation requires a matched set of verifiable source conflicts and a
dedicated rerun.

Single-expert labels are useful judgments, but do not by themselves establish
inter-annotator reliability or an error-free gold standard. Record the rubric
version and consider a second annotator or blinded re-annotation subset.
