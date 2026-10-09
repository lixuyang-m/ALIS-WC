"""Translate one natural-language instruction into scheduling constraints."""

import argparse
import json
from pathlib import Path

from .common import ROOT, parse_json, to_worker_clauses, validate_environment, valid_clauses, write_json
from .model import add_model_arguments, from_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--text", required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--num-depots", type=int, required=True)
    parser.add_argument("--system-prompt", type=Path, default=ROOT / "system_prompt.txt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        environment = {"num_agents": args.num_agents, "num_tasks": args.num_tasks, "num_depots": args.num_depots}
        validate_environment(environment)
        if args.max_new_tokens <= 0:
            raise ValueError("max-new-tokens must be positive")
        if args.output and args.output.exists():
            raise ValueError("Output already exists; choose a new path")
        system = args.system_prompt.read_text(encoding="utf-8").strip()
        user = json.dumps({"environment": environment, "text": args.text}, ensure_ascii=False)
        generator = from_arguments(args)
        result = generator.generate(system, user, args.max_new_tokens)
        try:
            clauses = parse_json(result["pred"])
            parse_ok = True
        except ValueError:
            clauses, parse_ok = None, False
        schema_ok = parse_ok and valid_clauses(clauses, environment)
        result.update({"mode": "model_inference", "model_path": args.model_path,
                       "adapter_path": args.adapter_path, "parse_ok": parse_ok,
                       "schema_ok": schema_ok, "clauses": clauses,
                       "worker_clauses": to_worker_clauses(clauses, environment) if schema_ok else None})
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
