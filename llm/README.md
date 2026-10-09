# Natural-language to LTL

Train a Qwen3-8B translator with full-parameter or LoRA fine-tuning, load a model
to translate new instructions, or evaluate the bundled prediction records.
Run the commands below from the repository root.

## Evaluate saved predictions

This entry point uses only the Python standard library. It reads saved model
outputs and recomputes metrics against the selected dataset; no GPU or model
loading is involved.

```bash
# List the available models and test splits.
python -m llm.evaluate_records --list

# Evaluate the Qwen3-8B full-parameter model's recorded outputs on both splits.
python -m llm.evaluate_records --model qwen3-8b-full

# Evaluate all 14 model/split combinations and save summaries and error cases.
python -m llm.evaluate_records --output-dir outputs/llm/recorded-evaluation
```

Available record names are `qwen3-8b-full`, `qwen3-8b-lora`, `qwen3-4b`,
`qwen3-30b`, `deepseek-r1-8b`, `deepseek-r1-14b`, and `mistral-small-3.1-24b`.
Use `--split simple`, `--split hard`, or `--split both` (default).
The Qwen3-8B full-parameter records yield 1,990/2,000 exact matches on simple
and 1,848/2,000 on hard.

`summary.json` contains counts and rates. Each `*.errors.jsonl` includes the
instruction, reference answer, predicted answer, and newly computed scores for
examples that do not exactly match. Existing output directories are not
overwritten.

Custom results are supported too:

```bash
python -m llm.evaluate_records \
  --data llm/data/qa_test_2000_simple.jsonl.gz \
  --predictions outputs/llm/simple.jsonl \
  --output-dir outputs/llm/simple-record-review
```

Each prediction must include a zero-based `global_idx` or `idx` into the selected
dataset and a `pred`, `pred_json`, or `pred_raw` value. File ordering does not
matter. If reference inputs, answers, or metadata are present, they are checked
against the dataset. Duplicate indices are rejected; use a single merged file.
Partial results require `--allow-partial`, and the summary reports coverage.
Saved correctness flags are always recomputed.

## Install model dependencies

Use a separate Python 3.10+ environment for LLM training and inference:

```bash
conda create -n alis-wc-llm python=3.10 -y
conda activate alis-wc-llm
pip install -r llm/requirements.txt
```

The model tools use Hugging Face Transformers, Accelerate, and PEFT. Install a
PyTorch build appropriate for your CUDA environment when using a GPU.

## Train

The bundled training dataset contains 6,000 conversations. A base model can be
specified using a Hugging Face model ID or a local directory.

```bash
# LoRA fine-tuning.
python -m llm.train \
  --model-path Qwen/Qwen3-8B \
  --config llm/configs/lora.json \
  --output-dir outputs/llm/qwen3-8b-lora

# Full-parameter fine-tuning on 8 GPUs with ZeRO-3.
pip install deepspeed
torchrun --standalone --nproc_per_node=8 -m llm.train \
  --model-path Qwen/Qwen3-8B \
  --config llm/configs/full.json \
  --deepspeed llm/configs/zero3.json \
  --output-dir outputs/llm/qwen3-8b-full
```

Full-parameter training can also use `python -m llm.train` without `--deepspeed`
when the model, gradients, and optimizer fit on the selected device. LoRA can
use `torchrun` for multiple GPUs. GPU count is chosen by the launcher.

| Setting | Full parameters | LoRA |
|---|---:|---:|
| Epochs | 1 | 1 |
| Batch size per device | 1 | 1 |
| Gradient accumulation | 4 | 4 |
| Learning rate | 1e-5 | 1e-4 |
| Maximum sequence length | 8,192 | 2,048 |
| Pack complete examples | Yes | No |
| Warmup ratio | 0.05 | 0.05 |
| LoRA rank / alpha | — | 8 / 32 |
| LoRA target modules | — | All linear layers |

Training uses cosine learning-rate decay and gradient checkpointing. Only the
assistant answer and EOS contribute to the loss; system/user tokens and padding
are masked. Packing concatenates complete conversations separated by EOS.
Examples exceeding `--max-length` produce an error rather than truncating the
reference answer. Training does not automatically use either test split for
validation or checkpoint selection.

Override common settings with `--learning-rate`, `--epochs`, `--batch-size`,
`--gradient-accumulation-steps`, and `--max-length`. For a short execution check,
add `--max-samples 8 --max-steps 1`. Use `--train-data` for another JSONL or
JSONL.GZ file with the same conversation format. `--device cpu --dtype float32`
is available for small-model debugging.

The output directory contains the trained model or LoRA adapter, tokenizer,
`recipe.json`, Trainer state, and epoch checkpoints. Use a new output directory
for each training run. A LoRA adapter is loaded together with its original base model.

## Translate one instruction

```bash
python -m llm.infer \
  --model-path outputs/llm/qwen3-8b-full \
  --num-agents 12 --num-tasks 40 --num-depots 6 \
  --text 'Robot 2 cannot visit Task 5; Task 3 must be completed before Task 8'
```

For a LoRA adapter, use the base model plus `--adapter-path`:

```bash
python -m llm.infer \
  --model-path Qwen/Qwen3-8B \
  --adapter-path outputs/llm/qwen3-8b-lora \
  --num-agents 12 --num-tasks 40 --num-depots 6 \
  --text 'Robot 2 cannot visit Task 5; Task 3 must be completed before Task 8'
```

The response includes the prediction text, parsed clauses, JSON/schema checks,
generation time, and scheduler-compatible `worker_clauses` when the schema is
valid. The expected clauses for this example are:

```json
[
  {"type": "safety", "agent_id": 2, "node_id": 11},
  {"type": "sequential", "predecessor_task_id": 3, "successor_task_id": 8}
]
```

IDs are zero-based. Depot `d` has node ID `d`; task `t` has node ID
`num_depots + t`. Sequential clauses use task IDs without a depot offset.
The scheduler representations are `(0, agent_id, node_id)` and
`(1, predecessor_task_id, successor_task_id)`.

`--model-path` accepts a Hugging Face directory containing model weights,
`config.json`, tokenizer files, and a chat template. `--device auto` uses available
GPUs and allows model sharding; a specific device such as `cuda:0` or `cpu` may
also be selected. `--local-files-only` prevents model downloads. Generation is
deterministic, with Qwen thinking disabled and `--max-new-tokens 1024` by default.
Use `--output <new-file.json>` to save the result.

## Evaluate a loaded model

```bash
python -m llm.evaluate \
  --model-path outputs/llm/qwen3-8b-full \
  --split simple \
  --output outputs/llm/simple.jsonl

python -m llm.evaluate \
  --model-path Qwen/Qwen3-8B \
  --adapter-path outputs/llm/qwen3-8b-lora \
  --split hard \
  --output outputs/llm/lora-hard.jsonl
```

Use `--limit 10` for a quick check or `--data <file.jsonl>` for a custom test set.
Predictions are saved per example, and a companion `<output>.summary.json`
contains the aggregate scores and model settings. Outputs must be new files.

Both evaluation modes use the same metrics:

| Field | Meaning |
|---|---|
| `parse_ok` | Output parses as strict JSON |
| `schema_ok` | Output is a list of supported clauses with integer IDs in range |
| `norm_match` | Exact JSON match after normalizing object keys and whitespace; clause order matters |
| `order_inv_match` | Exact multiset match of clauses; order does not matter, duplicate counts do |

Rates are fractions between 0 and 1. Schema validity checks syntax and ID ranges,
not whether the instruction was translated correctly. Model evaluation also
measures synchronized generation time per sample after `--warmup 1` untimed
generation. It excludes model loading and input tokenization. Recorded-result
evaluation computes accuracy metrics only.

## Files and verification

`data/` contains the 6,000 training, 2,000 simple-test, and 2,000 hard-test
conversations. `records/` contains one merged prediction file per model and
split. Files use lossless gzip compression and are read directly by the tools.
`manifest.json` records row counts and SHA-256 checksums of the decompressed
files. `system_prompt.txt` supplies the input/output specification used for
single-instruction inference.

```bash
# Standard-library checks: bundled data integrity, scoring, and index alignment.
python -m unittest llm.tests.test_records -v

# Also test full/LoRA training, model reload, inference, and evaluation on a
# locally constructed tiny Qwen3 model, using CPU and no downloads.
python -m unittest discover -s llm/tests -v
```

API references: [Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer),
[Qwen3](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3), and
[PEFT LoRA](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/lora).
