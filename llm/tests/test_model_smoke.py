"""CPU integration tests with a randomly initialized tiny Qwen3; no downloads."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from llm.common import ROOT, load_samples, write_jsonl
from llm.tests.test_records import record
from llm.train import SupervisedCollator, encode_samples

HAS_MODEL_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "accelerate"))


@unittest.skipUnless(HAS_MODEL_DEPS, "Install llm/requirements.txt for model smoke tests")
class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.base = cls.root / "base"
        cls.base.mkdir()
        vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "system": 3, "user": 4,
                 "assistant": 5, "type": 6, "safety": 7, "sequential": 8, "[": 9, "]": 10}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                pad_token="[PAD]", eos_token="[EOS]")
        cls.tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}"
                                       "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}")
        cls.tokenizer.save_pretrained(cls.base)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab), hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, max_position_embeddings=2048,
            eos_token_id=2, pad_token_id=1))
        model.save_pretrained(cls.base)
        cls.data = cls.root / "data.jsonl"
        write_jsonl(cls.data, [record(1), record(2)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_cli(self, module, *args):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT.parent,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_masking_packing_and_overlength(self):
        samples = load_samples(self.data)
        examples = encode_samples(samples, self.tokenizer, 512)
        self.assertEqual(examples[0]["labels"][-1], self.tokenizer.eos_token_id)
        self.assertIn(-100, examples[0]["labels"])
        packed = encode_samples(samples, self.tokenizer, 1024, packing=True)
        self.assertEqual(len(packed), 1)
        # Even when PAD and EOS share an ID, the real EOS must remain supervised.
        collator = SupervisedCollator(self.tokenizer.eos_token_id)
        short = {k: v[:-2] for k, v in examples[0].items()}
        batch = collator([examples[0], short])
        self.assertEqual(batch["labels"][0, -1].item(), self.tokenizer.eos_token_id)
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        with self.assertRaisesRegex(ValueError, "increase --max-length"):
            encode_samples(samples, self.tokenizer, 2)

    def test_full_and_lora_train_reload_evaluate(self):
        for method in ("full", "lora"):
            with self.subTest(method=method):
                trained = self.root / method
                self.run_cli("llm.train", "--model-path", self.base, "--config", ROOT / f"configs/{method}.json",
                             "--train-data", self.data, "--output-dir", trained, "--max-steps", 1,
                             "--gradient-accumulation-steps", 1, "--max-length", 512,
                             "--device", "cpu", "--dtype", "float32", "--local-files-only")
                self.assertTrue((trained / "trainer_state.json").is_file())
                model_args = ["--model-path", trained] if method == "full" else ["--model-path", self.base, "--adapter-path", trained]
                model_args += ["--device", "cpu", "--local-files-only", "--max-new-tokens", 4]
                result = self.run_cli("llm.infer", *model_args, "--text", "No constraints.",
                                     "--num-agents", 3, "--num-tasks", 5, "--num-depots", 2)
                self.assertEqual(json.loads(result.stdout)["mode"], "model_inference")
                predictions = self.root / f"{method}.jsonl"
                live = self.run_cli("llm.evaluate", *model_args, "--data", self.data,
                                    "--output", predictions, "--warmup", 0)
                replay = self.run_cli("llm.evaluate_records", "--data", self.data, "--predictions", predictions)
                self.assertEqual(json.loads(live.stdout)["norm_match"],
                                 json.loads(replay.stdout)["custom"]["norm_match"])


if __name__ == "__main__":
    unittest.main()
