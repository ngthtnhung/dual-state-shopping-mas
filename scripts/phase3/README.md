# Phase 3: simulated users and baseline comparison

Phase 3 runs five systems over the same deterministic, held-out VACOS/catalog
cases and generates user-facing answers with a local Ollama model. It uses
Python's standard library for Ollama HTTP calls; it does not send prompts to a
remote API. The default local model is `llama3:latest`.

Run a short smoke benchmark:

```powershell
python -m scripts.phase3.benchmark --sample-size 160 --cases-per-stratum 1 --top-k 10
```

Run the planned benchmark with 10 cases per stratum:

```powershell
python -m scripts.phase3.benchmark --sample-size 160 --cases-per-stratum 10 --seed 42
```

Use `--model`, `--ollama-host`, `--timeout`, and `--model-dir` to choose an
installed local Ollama model, adjust the local service, or locate the Phase 1
SetFit classifier. The endpoint must be loopback (`localhost`, `127.0.0.1`, or
`::1`). Check that the selected model is available with `ollama list` before
running. Install the Phase 1 Python dependencies into the **same interpreter**
used to launch the benchmark:

```powershell
python -m pip install -r requirements.txt
```

The benchmark checks for SetFit and spaCy before constructing its temporary
catalog index. If preparation fails after index creation, it closes SQLite
before propagating the error. Generation failures are reported; the evaluator
does not substitute mock answers.

## Frozen evaluation set

The evaluator samples the VACOS queries deterministically with the Phase 2
sampling code, uses the fixed Phase 2 thresholds (`U_Q=0.5`, metadata
coverage `0.5`, entropy `0.3`), and selects the same number of S1–S4 cases.
S3/S4 are controlled counterfactuals produced by masking target attributes in
the catalog. Each run writes:

- `reports/phase3/phase3_frozen_cases.jsonl`: exact query IDs, strata, parsed
  constraints, candidate parent ASINs, coverage/entropy, and counterfactual
  status.
- `reports/phase3/phase3_baseline_results.json`: input dataset SHA-256 hashes,
  frozen case-manifest hash, simulator evidence provenance, per-baseline
  metrics, routes, generated responses, token use, and latency.
- `reports/phase3/phase3_baseline_summary.csv`: one row per system and case
  for spreadsheet analysis.

Run all methods with the same seed, input paths, sample size, model, and cases
per stratum when comparing results. The hash manifest lets later runs verify
that frozen inputs and case selection have not drifted.

## Baselines

| ID | Implemented behavior |
|---|---|
| B1 | One local LLM response from the query alone; no catalog/review retrieval. |
| B2 | One Ollama ReAct-style agent plans a required catalog search and optional review search in structured JSON, consumes those tool results, then generates a response. |
| B3 | Fixed sequential path runs metadata evidence, review evidence, conflict evaluation, and decision synthesis for every case. |
| B4 | Query-complexity-only routing; it does not use catalog completeness to decide whether to escalate or clarify. |
| Ours | Phase 2 dual-conditioned orchestrator consumes both query state and the frozen catalog state; evidence dispatch is policy-controlled. |

All systems use the same local Ollama model for response generation; B2 also
uses it for its structured tool-planning turn. The deterministic policy/tool
work is separately counted from LLM generation calls and tokens. A clarification is passed to the same simulated user; cooperative
users receive one generated follow-up response, while guarded users answer
evasively and non-cooperative users decline.

## Review-grounded simulated users

For the tested candidate products, the script scans a bounded prefix of the
filtered Amazon laptop review file, keeps up to six reviews per candidate, and
classifies detected aspect spans with the Phase 1 SetFit ABSA model. A
deterministic seed selects actual review sentences and sentiment labels as
synthetic preference anchors. VACOS vague annotations provide an
explicit/ambiguous profile; cooperation is assigned reproducibly among
cooperative, guarded, and non-cooperative profiles. Personas and answers are
synthetic: they are not the review authors and are not an exact reproduction
of Tran et al. or Petruzzelli et al. The cited works' full simulation
protocols were not available in the project materials.

## Metrics and limitations

The report includes expected-route rate for S1–S4, clarification and
abstention rates, review-agent activation, tool/agent/generation calls,
Ollama prompt/output token counts, and end-to-end latency. The automated
expected-route score measures routing-policy compliance, not recommendation
quality. No gold recommendation, evidence precision, unsupported-claim, or
user-satisfaction labels are available in these data. Generated answers and
citations still require human annotation before making factuality or quality
claims. S3/S4 rows are counterfactual catalog conditions, not natural missing
metadata examples. The evaluation uses VACOS queries where the current parser
recognizes core attributes; it is not representative of the full user query
distribution. Per-case latency starts at baseline routing and includes Ollama
generation/follow-up, but excludes shared one-time catalog indexing, query
profiling, and SetFit review-index construction; those setup costs are not
directly comparable with B1, which does not retrieve evidence.

B1 does not expose a deterministic system action, so its expected-route score
is reported as not applicable rather than treating direct generation as a
failed routing decision.

Run tests with:

```powershell
python -m unittest discover -s tests -v
```
