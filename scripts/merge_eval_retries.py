"""Fill missing Ragas cells from retry reports without overwriting valid scores."""

import argparse
import csv
import json
import math
from pathlib import Path


def valid(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--retry", action="append", required=True)
    parser.add_argument("--retry-judge", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    with open(args.base, encoding="utf-8") as handle:
        report = json.load(handle)
    metrics = report["metrics"]
    by_question = {row.get("question"): row for row in report["scores"]}
    fills = []

    for retry_path in args.retry:
        with open(retry_path, encoding="utf-8") as handle:
            retry = json.load(handle)
        for retry_row in retry.get("scores", []):
            question = retry_row.get("question")
            base_row = by_question.get(question)
            if base_row is None:
                continue
            filled_metrics = []
            for metric in metrics:
                if not valid(base_row.get(metric)) and valid(retry_row.get(metric)):
                    base_row[metric] = round(float(retry_row[metric]), 4)
                    filled_metrics.append(metric)
            if filled_metrics:
                fills.append({
                    "index": report["scores"].index(base_row) + 1,
                    "question": question,
                    "metrics": filled_metrics,
                    "judge": args.retry_judge,
                    "source": str(Path(retry_path).resolve()),
                })

    report["valid_counts"] = {
        metric: sum(valid(row.get(metric)) for row in report["scores"])
        for metric in metrics
    }
    report["average"] = {}
    for metric in metrics:
        values = [row[metric] for row in report["scores"] if valid(row.get(metric))]
        report["average"][metric] = round(sum(values) / len(values), 4) if values else 0.0
    report["complete_count"] = sum(
        all(valid(row.get(metric)) for metric in metrics)
        for row in report["scores"]
    )
    report["composite"] = round(
        sum(report["average"][metric] for metric in metrics) / len(metrics), 4
    )

    prior = report.get("evaluation_config", {}).get("judge_model", "unknown")
    report["evaluation_config"] = {
        **report.get("evaluation_config", {}),
        "judge_model": "mixed",
        "primary_judge_model": prior,
        "retry_judge_model": args.retry_judge,
        "mixed_judge": True,
        "note": "Existing valid scores were preserved; only missing cells were filled by the retry judge.",
    }
    report["retry_fills"] = fills
    report["metric_judge_counts"] = {
        metric: {
            prior: report["valid_counts"][metric]
                   - sum(metric in fill["metrics"] for fill in fills),
            args.retry_judge: sum(metric in fill["metrics"] for fill in fills),
        }
        for metric in metrics
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", *metrics])
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in ["question", *metrics]}
            for row in report["scores"]
        )
        writer.writerow({"question": "__average__", **report["average"]})

    print(json.dumps({
        "filled_samples_count": len(fills),
        "filled_metric_cells": sum(len(fill["metrics"]) for fill in fills),
        "filled_samples": sorted({fill["index"] for fill in fills}),
        "valid_counts": report["valid_counts"],
        "complete_count": report["complete_count"],
        "average": report["average"],
        "composite": report["composite"],
        "out_json": str(out_json.resolve()),
        "out_csv": str(Path(args.out_csv).resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
