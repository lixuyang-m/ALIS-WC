"""Fine-tune an instruction model with full parameters or LoRA."""

import argparse
import json
import sys
from pathlib import Path

from .common import DATASETS, ROOT, load_samples


def encode_samples(samples, tokenizer, max_length, packing=False):
    """Supervise assistant JSON and EOS only; reject overlong examples intact."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Training requires an EOS token")
    sequences = []
    for sample in samples:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sample.system}, {"role": "user", "content": sample.user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt = tokenizer.encode(text, add_special_tokens=False)
        answer = tokenizer.encode(sample.gold, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt) + len(answer) > max_length:
            raise ValueError(f"Sample {sample.id} needs {len(prompt) + len(answer)} tokens; "
                             f"increase --max-length (currently {max_length})")
        sequences.append({"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer})
    if not packing:
        return sequences
    packed, current = [], {"input_ids": [], "labels": []}
    for example in sequences:
        if current["input_ids"] and len(current["input_ids"]) + len(example["input_ids"]) > max_length:
            packed.append(current)
            current = {"input_ids": [], "labels": []}
        for key in current:
            current[key].extend(example[key])
    if current["input_ids"]:
        packed.append(current)
    return packed


class SupervisedCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in features:
            n = len(example["input_ids"])
            batch["input_ids"].append(example["input_ids"] + [self.pad_token_id] * (length - n))
            batch["attention_mask"].append([1] * n + [0] * (length - n))
            batch["labels"].append(example["labels"] + [-100] * (length - n))
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B", help="Base model directory or Hugging Face ID")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lora.json")
    parser.add_argument("--train-data", type=Path, default=DATASETS["train"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepspeed", type=Path, help="Optional DeepSpeed JSON configuration")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Limit training data for a smoke test")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        for key in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate", "max_length"):
            value = getattr(args, key)
            if value is not None:
                config[key] = value
            if config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if config.get("method") not in ("full", "lora"):
            raise ValueError("Config method must be full or lora")
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        if args.max_steps == 0 or args.max_steps < -1:
            raise ValueError("max-steps must be -1 or a positive integer")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Output directory is not empty; choose a new directory")
        samples = load_samples(args.train_data)[:args.max_samples]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r llm/requirements.txt") from exc
        use_cpu = args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available())
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        dtype = args.dtype
        if dtype == "auto":
            dtype = "float32" if use_cpu else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        if use_cpu and dtype == "float16":
            raise ValueError("Use float32 or bfloat16 for CPU training")
        set_seed(config["seed"])
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
        if not tokenizer.chat_template:
            raise ValueError("The tokenizer must provide a chat_template")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        dataset = encode_samples(samples, tokenizer, config["max_length"], config.get("packing", False))
        # Build TrainingArguments before loading the model so ZeRO-3 can initialize it sharded.
        training_args = TrainingArguments(
            output_dir=str(args.output_dir), num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"], warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type="cosine", max_steps=args.max_steps,
            logging_steps=config["logging_steps"], save_strategy="epoch",
            save_total_limit=config["save_total_limit"], report_to=[],
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=dtype == "bfloat16", fp16=dtype == "float16", use_cpu=use_cpu,
            deepspeed=str(args.deepspeed) if args.deepspeed else None,
            seed=config["seed"], data_seed=config["seed"],
            ddp_find_unused_parameters=False, remove_unused_columns=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=getattr(torch, dtype),
            attn_implementation=args.attn_implementation, local_files_only=args.local_files_only,
        )
        model.config.use_cache = False
        if config["method"] == "lora":
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
            ))
            model.enable_input_require_grads()
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                          data_collator=SupervisedCollator(tokenizer.pad_token_id), processing_class=tokenizer)
        if trainer.is_world_process_zero():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            recipe = {"config": config, "model_path": args.model_path,
                      "train_data": str(args.train_data), "training_samples": len(samples),
                      "training_sequences": len(dataset), "dtype": dtype,
                      "max_steps": args.max_steps, "attn_implementation": args.attn_implementation,
                      "deepspeed": str(args.deepspeed) if args.deepspeed else None}
            (args.output_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
            print(f"Training {config['method']}: {len(samples)} examples, {len(dataset)} sequences", file=sys.stderr)
        trainer.train()
        model.config.use_cache = True
        trainer.save_model(str(args.output_dir))
        trainer.save_state()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(str(args.output_dir))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
