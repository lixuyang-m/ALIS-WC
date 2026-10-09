"""Data loading and scoring shared by live and recorded evaluations (stdlib only)."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "train": ROOT / "data/qa_train_6000.jsonl.gz",
    "simple": ROOT / "data/qa_test_2000_simple.jsonl.gz",
    "hard": ROOT / "data/qa_test_2000_harder.jsonl.gz",
}


def parse_json(text):
    def reject(value):
        raise ValueError(f"Non-JSON constant: {value}")
    return json.loads(text, parse_constant=reject)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = parse_json(line)
                    if not isinstance(row, dict):
                        raise ValueError("Expected an object")
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
                yield row


@dataclass
class Sample:
    id: int
    system: str
    user: str
    gold: str
    meta: dict


def validate_environment(env):
    names = {"num_agents", "num_tasks", "num_depots"}
    if not isinstance(env, dict) or set(env) != names:
        raise ValueError(f"environment must contain exactly {sorted(names)}")
    if any(type(env[k]) is not int or env[k] < 0 for k in names):
        raise ValueError("Environment counts must be non-negative integers")


def valid_clauses(clauses, environment):
    if not isinstance(clauses, list):
        return False
    for clause in clauses:
        if not isinstance(clause, dict):
            return False
        kind = clause.get("type")
        if kind == "safety":
            fields = {"type", "agent_id", "node_id"}
            limits = {"agent_id": environment["num_agents"],
                      "node_id": environment["num_depots"] + environment["num_tasks"]}
        elif kind == "sequential":
            fields = {"type", "predecessor_task_id", "successor_task_id"}
            limits = {"predecessor_task_id": environment["num_tasks"],
                      "successor_task_id": environment["num_tasks"]}
        else:
            return False
        if set(clause) != fields:
            return False
        if any(type(clause[k]) is not int or not 0 <= clause[k] < n for k, n in limits.items()):
            return False
    return True


def load_samples(path):
    samples = []
    ids = set()
    for index, row in enumerate(read_jsonl(path)):
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Sample {index}: expected system/user/assistant messages")
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"Sample {index}: message content must be text")
        user = parse_json(messages[1]["content"])
        validate_environment(user.get("environment"))
        if not isinstance(user.get("text"), str):
            raise ValueError(f"Sample {index}: text must be a string")
        gold = parse_json(messages[2]["content"])
        if not valid_clauses(gold, user["environment"]):
            raise ValueError(f"Sample {index}: invalid reference clauses")
        sample_id = row.get("id", index)
        if type(sample_id) is not int or sample_id in ids:
            raise ValueError(f"Sample {index}: invalid or duplicate sample id")
        ids.add(sample_id)
        samples.append(Sample(sample_id, messages[0]["content"], messages[1]["content"],
                              messages[2]["content"], row.get("meta", {})))
    if not samples:
        raise ValueError(f"Empty dataset: {path}")
    return samples


def score_prediction(prediction, sample):
    gold = parse_json(sample.gold)
    environment = parse_json(sample.user)["environment"]
    try:
        pred = parse_json(prediction) if isinstance(prediction, str) else prediction
        # Historical predictions are text or an already parsed JSON value.
        normalized = canonical(pred)
    except (ValueError, TypeError):
        return {"parse_ok": False, "schema_ok": False, "norm_match": False, "order_inv_match": False}
    ordered = normalized == canonical(gold)
    unordered = (isinstance(pred, list) and all(isinstance(c, dict) for c in pred)
                 and Counter(canonical(c) for c in pred) == Counter(canonical(c) for c in gold))
    return {"parse_ok": True, "schema_ok": valid_clauses(pred, environment),
            "norm_match": ordered, "order_inv_match": unordered}


def summarize(scored, dataset_size):
    if not scored:
        raise ValueError("No predictions to evaluate")
    n = len(scored)
    result = {"samples": n, "dataset_samples": dataset_size, "coverage": n / dataset_size}
    for key in ("parse_ok", "schema_ok", "norm_match", "order_inv_match"):
        count = sum(bool(r[key]) for r in scored)
        result[key] = {"correct": count, "total": n, "rate": count / n}
    return result


def to_worker_clauses(clauses, environment):
    if not valid_clauses(clauses, environment):
        raise ValueError("Prediction contains invalid clauses or out-of-range IDs")
    return [(0, c["agent_id"], c["node_id"]) if c["type"] == "safety" else
            (1, c["predecessor_task_id"], c["successor_task_id"]) for c in clauses]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
