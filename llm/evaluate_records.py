"""Recompute metrics from saved predictions without loading a model."""

import argparse
import json
from pathlib import Path

from .common import (ROOT, canonical, load_samples, parse_json, read_jsonl,
                     score_prediction, summarize, write_json, write_jsonl)


def evaluate_records(data_path, predictions_path, allow_partial=False):
    samples = load_samples(data_path)
    seen, scored, errors = set(), [], []
    for row in read_jsonl(predictions_path):
        idx = row.get("global_idx", row.get("idx"))
        if "idx" in row and "global_idx" in row and row["idx"] != row["global_idx"]:
            raise ValueError("Conflicting idx/global_idx fields")
        if type(idx) is not int or not 0 <= idx < len(samples):
            raise ValueError(f"Invalid prediction index: {idx!r}")
        if idx in seen:
            raise ValueError(f"Duplicate prediction index: {idx}; use one merged result file")
        seen.add(idx)
        sample = samples[idx]
        for key, expected in (("user", sample.user), ("gold", sample.gold)):
            if key in row:
                actual = parse_json(row[key]) if isinstance(row[key], str) else row[key]
                if canonical(actual) != canonical(parse_json(expected)):
                    raise ValueError(f"Prediction {idx}: {key} does not match the selected dataset")
        if "meta" in row and row["meta"] != sample.meta:
            raise ValueError(f"Prediction {idx}: metadata does not match the selected dataset")
        field = next((k for k in ("pred", "pred_json", "pred_raw") if k in row), None)
        if field is None:
            raise ValueError(f"Prediction {idx}: no pred, pred_json, or pred_raw field")
        score = score_prediction(row[field], sample)
        scored.append(score)
        if not score["norm_match"]:
            errors.append({"global_idx": idx, "sample_id": sample.id, "user": sample.user,
                           "gold": sample.gold, "pred": row[field], **score})
    if not allow_partial and len(seen) != len(samples):
        raise ValueError(f"Incomplete predictions: {len(seen)}/{len(samples)}; use --allow-partial explicitly")
    return {"mode": "recorded_predictions", **summarize(scored, len(samples))}, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List bundled model/split combinations")
    parser.add_argument("--model", default="all", help="Bundled model name, or all")
    parser.add_argument("--split", choices=["simple", "hard", "both"], default="both")
    parser.add_argument("--predictions", type=Path, help="Evaluate a custom JSONL or JSONL.GZ result file")
    parser.add_argument("--data", type=Path, help="Reference dataset for --predictions")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="New directory for summary.json and error cases")
    args = parser.parse_args()
    try:
        manifest = json.loads((ROOT / "manifest.json").read_text())
        records = manifest["records"]
        if args.list:
            print(json.dumps([{k: r[k] for k in ("model", "split", "path")} for r in records], indent=2))
            return
        if bool(args.predictions) != bool(args.data):
            raise ValueError("--predictions and --data must be provided together")
        if args.predictions:
            selected = [("custom", args.data, args.predictions)]
        else:
            selected = [(r["model"] + "_" + r["split"], ROOT / manifest["datasets"][r["split"]]["path"], ROOT / r["path"])
                        for r in records if (args.model == "all" or r["model"] == args.model)
                        and (args.split == "both" or r["split"] == args.split)]
        if not selected:
            raise ValueError("No matching records; use --list to see available models")
        if args.output_dir and args.output_dir.exists():
            raise ValueError("--output-dir already exists; choose a new directory")
        summaries, error_sets = {}, {}
        for name, data, predictions in selected:
            summary, errors = evaluate_records(data, predictions, args.allow_partial)
            summaries[name], error_sets[name] = summary, errors
        if args.output_dir:
            write_json(args.output_dir / "summary.json", summaries)
            for name, errors in error_sets.items():
                write_jsonl(args.output_dir / f"{name}.errors.jsonl", errors)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
