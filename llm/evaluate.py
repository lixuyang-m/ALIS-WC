"""Generate model predictions for a dataset and compute NL-to-LTL metrics."""

import argparse
import json
import sys
from pathlib import Path

from .common import DATASETS, load_samples, score_prediction, summarize, write_json
from .model import add_model_arguments, from_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--split", choices=["simple", "hard"], default="simple")
    source.add_argument("--data", type=Path, help="Custom system/user/assistant JSONL(.gz) dataset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; a .summary.json is saved alongside it")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed generations before evaluation")
    args = parser.parse_args()
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        if args.warmup < 0 or args.max_new_tokens <= 0:
            raise ValueError("warmup must be non-negative and max-new-tokens positive")
        summary_path = Path(str(args.output) + ".summary.json")
        if args.output.exists() or summary_path.exists():
            raise ValueError("Output already exists; choose a new path")
        data_path = args.data or DATASETS[args.split]
        all_samples = load_samples(data_path)
        samples = all_samples[:args.limit]
        generator = from_arguments(args)
        for _ in range(args.warmup):
            generator.generate(samples[0].system, samples[0].user, args.max_new_tokens)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored, elapsed, tokens = [], 0.0, 0
        with args.output.open("x", encoding="utf-8") as stream:
            for idx, sample in enumerate(samples):
                prediction = generator.generate(sample.system, sample.user, args.max_new_tokens)
                score = score_prediction(prediction["pred"], sample)
                scored.append(score)
                elapsed += prediction["generation_seconds"]
                tokens += prediction["generated_tokens"]
                row = {"global_idx": idx, "sample_id": sample.id, "meta": sample.meta,
                       "user": sample.user, "gold": sample.gold, **prediction, **score}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if (idx + 1) % 50 == 0 or idx + 1 == len(samples):
                    print(f"Evaluated {idx + 1}/{len(samples)}", file=sys.stderr)
        summary = {"mode": "model_evaluation", "model_path": args.model_path,
                   "adapter_path": args.adapter_path, "data": str(data_path),
                   "max_new_tokens": args.max_new_tokens, "warmup": args.warmup,
                   **summarize(scored, len(all_samples)),
                   "generation_seconds_total": elapsed,
                   "generation_seconds_mean": elapsed / len(samples),
                   "generated_tokens": tokens}
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
