from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("positive", "negative", "neutral", "conflict")


def _training_text(text: str, span: str) -> str:
    return f"Review: {text}\nAspect: {span}"


def _split_indices(
    labels: list[str], source_texts: list[str], seed: int
) -> tuple[Any, Any]:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(
        splitter.split(
            X=source_texts,
            y=labels,
            groups=source_texts,
        )
    )


def _validation_metrics(model: Any, validation_dataset: Any) -> dict[str, Any]:
    from sklearn.metrics import classification_report, f1_score

    labels = validation_dataset["label"]
    predictions = model.predict(validation_dataset["text"])
    metrics = {
        "accuracy": float(
            sum(
                actual == predicted
                for actual, predicted in zip(labels, predictions, strict=True)
            )
            / len(labels)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            labels,
            predictions,
            labels=list(LABELS),
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def train_model(dataset_dir: Path, output_dir: Path, base_model: str, seed: int) -> dict[str, Any]:
    try:
        from datasets import Dataset, load_from_disk
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as error:
        raise RuntimeError(
            "SetFit training dependencies are missing. Install "
            "`requirements.txt` in the active Python environment."
        ) from error

    dataset = load_from_disk(str(dataset_dir))
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Expected train and test splits in the saved DatasetDict")

    labeled_train = [
        row
        for row in dataset["train"]
        if row.get("label") in LABELS and row.get("text") and row.get("span")
    ]
    if not labeled_train:
        raise ValueError("No labeled SetFit training examples were found")
    missing_labels = set(LABELS) - {row["label"] for row in labeled_train}
    if missing_labels:
        raise ValueError(f"Training data is missing required labels: {sorted(missing_labels)}")

    examples = Dataset.from_dict(
        {
            "text": [_training_text(row["text"], row["span"]) for row in labeled_train],
            "label": [row["label"] for row in labeled_train],
        }
    )
    train_indices, validation_indices = _split_indices(
        labels=examples["label"],
        source_texts=[row["text"] for row in labeled_train],
        seed=seed,
    )
    train_dataset = examples.select(train_indices)
    validation_dataset = examples.select(validation_indices)

    model = SetFitModel.from_pretrained(base_model, labels=list(LABELS))
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        batch_size=16,
        num_epochs=1,
        num_iterations=20,
        seed=seed,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        metric="accuracy",
    )
    trainer.train()
    metrics = _validation_metrics(trainer.model, validation_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    unlabeled_test_examples = sum(
        not row.get("label") for row in dataset["test"]
    )
    summary = {
        "base_model": base_model,
        "labels": list(LABELS),
        "source_train_examples": len(dataset["train"]),
        "labeled_source_train_examples": sum(
            row.get("label") in LABELS for row in dataset["train"]
        ),
        "source_train_label_counts": {
            label: count
            for label, count in Counter(
                row["label"] for row in labeled_train
            ).items()
        },
        "source_train_examples_excluded_missing_text_or_span": (
            len(dataset["train"]) - len(labeled_train)
        ),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "validation_split": "5-fold stratified group split by source sentence text",
        "seed": seed,
        "unlabeled_source_test_examples_excluded": unlabeled_test_examples,
        "validation_metrics": metrics,
        "model_dir": str(output_dir),
    }
    with (output_dir / "phase1_training_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a four-label SetFit aspect-based sentiment classifier."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/semeval"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/setfit-absa-laptops")
    )
    parser.add_argument(
        "--base-model", default="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            train_model(args.dataset_dir, args.output_dir, args.base_model, args.seed),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
