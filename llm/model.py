"""Hugging Face model loading, shared by single-command and dataset inference."""

import time
from pathlib import Path


def add_model_arguments(parser):
    parser.add_argument("--model-path", required=True, help="Full model directory or Hugging Face model ID")
    parser.add_argument("--adapter-path", help="Optional LoRA adapter directory; model-path is the base model")
    parser.add_argument("--device", default="auto", help="auto (GPU sharding if available), cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")


class Generator:
    def __init__(self, model_path, adapter_path=None, device="auto", dtype="auto", local_files_only=False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the model dependencies with: pip install -r llm/requirements.txt") from exc
        self.torch = torch
        self.model_path, self.adapter_path = model_path, adapter_path
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu")
        kwargs = {"local_files_only": local_files_only}
        if dtype != "auto":
            kwargs["dtype"] = getattr(torch, dtype)
        else:
            kwargs["dtype"] = "auto" if torch.cuda.is_available() and device != "cpu" else torch.float32
        if device == "auto" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": "cpu" if device == "auto" else device}
        tokenizer_path = model_path
        if adapter_path and (Path(adapter_path) / "tokenizer_config.json").is_file():
            tokenizer_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only)
        if not self.tokenizer.chat_template:
            raise ValueError("The tokenizer must provide a chat_template")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=local_files_only)
        self.model.eval()

    def _synchronize(self):
        devices = {p.device for p in self.model.parameters() if p.device.type == "cuda"}
        for device in devices:
            self.torch.cuda.synchronize(device)

    def generate(self, system, user, max_new_tokens=1024):
        if max_new_tokens <= 0:
            raise ValueError("max-new-tokens must be positive")
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        self._synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self._synchronize()
        elapsed = time.perf_counter() - start
        tokens = output[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        prediction = raw.rsplit("</think>", 1)[-1].strip()
        return {"pred": prediction, "pred_raw": raw, "generation_seconds": elapsed,
                "generated_tokens": len(tokens)}


def from_arguments(args):
    return Generator(args.model_path, args.adapter_path, args.device, args.dtype, args.local_files_only)
