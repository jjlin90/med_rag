"""Fill missing Ragas cells from retry reports without overwriting valid scores."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def valid(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


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
    if report.get('engine') != 'ragas':
        raise ValueError('Base report must use the Ragas engine')
    metrics = report["metrics"]
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError('Metrics must be nonempty and unique')
    by_question = {row.get("question"): row for row in report["scores"]}
    if len(by_question) != len(report['scores']):
        raise ValueError('Base report contains duplicate questions')
    previous_fills = report.get('retry_fills', [])
    fills = []

    for retry_path in args.retry:
        with open(retry_path, encoding="utf-8") as handle:
            retry = json.load(handle)
        if retry.get('engine') != 'ragas':
            raise ValueError('Retry report must use the Ragas engine')
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
    raw_average = {}
    for metric in metrics:
        values = [row[metric] for row in report["scores"] if valid(row.get(metric))]
        raw_average[metric] = statistics.mean(values) if values else None
        report["average"][metric] = round(raw_average[metric], 4) if values else None
        for row in report['scores']:
            if not valid(row.get(metric)):
                row[metric] = None
    report["complete_count"] = sum(
        all(valid(row.get(metric)) for metric in metrics)
        for row in report["scores"]
    )
    # Round once, and only summarize a complete common sample set.
    complete_rows = [row for row in report['scores'] if all(valid(row.get(m)) for m in metrics)]
    report['composite'] = (round(statistics.mean(row[m] for row in complete_rows for m in metrics), 4)
                           if complete_rows else None)

    prior = report.get('evaluation_config', {}).get('primary_judge_model') or report.get("evaluation_config", {}).get("judge_model", "unknown")
    report["evaluation_config"] = {
        **report.get("evaluation_config", {}),
        "judge_model": "mixed",
        "primary_judge_model": prior,
        "retry_judge_model": args.retry_judge,
        "mixed_judge": True,
        "note": "Existing valid scores were preserved; only missing cells were filled by the retry judge.",
    }
    report["retry_fills"] = previous_fills + fills
    counts = {}
    for metric in metrics:
        metric_counts = {}
        for fill in report['retry_fills']:
            if metric in fill['metrics']:
                judge = fill['judge']
                metric_counts[judge] = metric_counts.get(judge, 0) + 1
        metric_counts[prior] = metric_counts.get(prior, 0) + report['valid_counts'][metric] - sum(metric_counts.values())
        counts[metric] = metric_counts
    report['metric_judge_counts'] = counts

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as handle:
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
