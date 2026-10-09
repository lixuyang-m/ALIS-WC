import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from llm.common import (ROOT, Sample, load_samples, score_prediction,
                        to_worker_clauses, write_jsonl)
from llm.evaluate_records import evaluate_records


def record(sample_id=1):
    user = {"environment": {"num_agents": 3, "num_tasks": 5, "num_depots": 2},
            "text": "Robot 1 cannot visit Task 2; Task 0 before Task 3"}
    gold = [{"type": "safety", "agent_id": 1, "node_id": 4},
            {"type": "sequential", "predecessor_task_id": 0, "successor_task_id": 3}]
    return {"id": sample_id, "meta": {"answer_id": sample_id}, "messages": [
        {"role": "system", "content": "Return a JSON array."},
        {"role": "user", "content": json.dumps(user)},
        {"role": "assistant", "content": json.dumps(gold)}]}


class RecordEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data.jsonl"
        write_jsonl(self.data, [record(1), record(2)])
        self.samples = load_samples(self.data)

    def predictions(self, rows):
        path = self.root / "predictions.jsonl"
        write_jsonl(path, rows)
        return path

    def test_historical_schemas_and_shuffled_indices(self):
        rows = [{"idx": 1, "pred_json": self.samples[1].gold, "match_norm": False},
                {"global_idx": 0, "pred": "invalid", "norm_match": True}]
        summary, errors = evaluate_records(self.data, self.predictions(rows))
        self.assertEqual(summary["norm_match"]["correct"], 1)
        self.assertEqual(errors[0]["global_idx"], 0)

    def test_duplicate_indices_rejected(self):
        row = {"idx": 0, "pred": self.samples[0].gold}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_records(self.data, self.predictions([row, row]))

    def test_missing_indices_need_explicit_partial_mode(self):
        path = self.predictions([{"idx": 0, "pred": self.samples[0].gold}])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluate_records(self.data, path)
        summary, _ = evaluate_records(self.data, path, allow_partial=True)
        self.assertEqual(summary["coverage"], 0.5)

    def test_wrong_dataset_rejected(self):
        path = self.predictions([{"idx": 0, "pred": "[]", "gold": "[]"}])
        with self.assertRaisesRegex(ValueError, "gold does not match"):
            evaluate_records(self.data, path)

    def test_order_invariance_preserves_duplicates(self):
        sample = self.samples[0]
        gold = json.loads(sample.gold)
        reversed_score = score_prediction(json.dumps(gold[::-1]), sample)
        self.assertFalse(reversed_score["norm_match"])
        self.assertTrue(reversed_score["order_inv_match"])
        self.assertFalse(score_prediction(json.dumps(gold + gold[:1]), sample)["order_inv_match"])

    def test_strict_json_and_schema_validation(self):
        sample = self.samples[0]
        self.assertFalse(score_prediction("NaN", sample)["parse_ok"])
        self.assertFalse(score_prediction("```json\n[]\n```", sample)["parse_ok"])
        self.assertTrue(score_prediction("null", sample)["parse_ok"])
        self.assertFalse(score_prediction("null", sample)["schema_ok"])
        for value in [True, -1, 3]:
            invalid = [{"type": "safety", "agent_id": value, "node_id": 4}]
            self.assertFalse(score_prediction(invalid, sample)["schema_ok"])

    def test_worker_conversion_keeps_depot_offset(self):
        sample = self.samples[0]
        self.assertEqual(to_worker_clauses(json.loads(sample.gold), json.loads(sample.user)["environment"]),
                         [(0, 1, 4), (1, 0, 3)])

    def test_bundled_bytes_and_all_prediction_files(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        for item in list(manifest["datasets"].values()) + manifest["records"]:
            with gzip.open(ROOT / item["path"], "rb") as stream:
                data = stream.read()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256_uncompressed"])
        for item in manifest["records"]:
            dataset = ROOT / manifest["datasets"][item["split"]]["path"]
            summary, _ = evaluate_records(dataset, ROOT / item["path"])
            self.assertEqual(summary["samples"], 2000)
            if item["model"] == "qwen3-8b-full":
                self.assertEqual(summary["norm_match"]["correct"], 1990 if item["split"] == "simple" else 1848)


if __name__ == "__main__":
    unittest.main()
