# Phase 1 data preparation

Run these commands from the repository root. Amazon JSONL files are processed as
streams; the review archive is read and written as gzip without extracting the
full Electronics file to disk.

```powershell
python scripts\phase1\prepare_amazon.py
python scripts\phase1\prepare_vacos_queries.py
```

Amazon outputs are written to `dataset/processed/amazon-laptops/`:

- `laptops_catalog.jsonl`: one metadata row per selected `parent_asin`.
- `laptop_parent_asins.txt`: keys used to link reviews.
- `laptops_reviews.jsonl.gz`: reviews whose `parent_asin` is in the laptop catalog; review `asin` is preserved to identify variants.
- `catalog_coverage.json`: keyword-based Display, RAM, Battery, and Weight/Portability coverage counts and percentages.

The laptop filter requires a category path component exactly matching
`Laptop`, `Laptops`, or `Laptop Computers`. Coverage measures whether metadata
text contains an attribute signal; it does not claim that a usable value was
extracted or verified.

VACOS v2 produces `dataset/processed/vacos_laptops_bio.jsonl`. It contains the
1,695 laptop queries, source token offsets, key-fact BIO tags, separately
derived vague-word BIO tags, and original annotation details. The source has
23 key-fact annotations without an attribute category; these are preserved
with `UNKNOWN` BIO labels and counted in the command summary. The counts are
also persisted in `dataset/processed/vacos_laptops_bio.summary.json`.

## SetFit ABSA training

The downloaded SemEval `DatasetDict` contains 2,358 labeled training examples,
but all 654 examples in its test split have empty labels. The trainer therefore
holds out a deterministic, stratified 20% validation fold from the labeled
training data, grouping by source sentence text to prevent aspects from the
same sentence leaking across the split.

Install the extra training dependencies in the Python environment used for
training, then run:

```powershell
python -m pip install -r requirements.txt
python scripts\phase1\train_setfit_absa.py
```

The classifier input combines the review text and target aspect. The trained
model and validation summary are written to `models/setfit-absa-laptops/`.
The recorded validation metrics include accuracy, macro-F1, and per-class
precision/recall/F1; interpret the `conflict` metrics cautiously because that
class has only 45 labeled source examples.

## Review ABSA Agent

`review_absa_agent.py` uses spaCy's English tokenizer and sentencizer plus a
configurable `PhraseMatcher` dictionary to find aspect spans in each review
sentence. It then classifies each `(sentence, aspect)` pair with the saved
SetFit model. The default dictionary covers Display, RAM, Battery, and
Weight/Portability. Common abbreviations such as `e.g.` and `i.e.` are kept
with the following sentence instead of causing a false split.

Run it from the repository root to stream the filtered reviews into a JSONL
file of linkable sentiment evidence:

```powershell
python -m scripts.phase1.review_absa_agent
```

By default, the input is `dataset/processed/amazon-laptops/laptops_reviews.jsonl.gz`
and the output is `dataset/processed/amazon-laptops/review_aspect_evidence.jsonl`.
Override paths with `--input`, `--output`, and `--model-dir`; tune inference
memory/use with `--batch-size`. Every evidence row preserves `parent_asin`,
variant `asin`, rating, review title and timestamp, as well as exact sentence
and aspect character offsets. The dictionary is a high-recall heuristic and
may include ambiguous terms such as `power`; inspect extracted spans before
treating model outputs as verified product facts.

The agent can also be embedded directly:

```python
from scripts.phase1.review_absa_agent import ReviewABSAAgent

agent = ReviewABSAAgent(model_dir="models/setfit-absa-laptops")
evidence_rows = agent.analyze_review(review_record)
```
