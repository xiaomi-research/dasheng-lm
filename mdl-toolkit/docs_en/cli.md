# Command Line Interface Reference

`mdl-toolkit` provides the following subcommands:

## `mdl-toolkit convert-dataset` — convert datasets

The `mdl-toolkit convert-dataset` command converts a CSV dataset into a Hugging Face Datasets format with all training-required fields, adds special tokens, tokenizes inputs, and produces training labels.

If a CSV dataset is passed to `mdl-toolkit train`, it will be converted before training. In that case, all options of `mdl-toolkit convert-dataset` (except input/output arguments) also apply to `mdl-toolkit train` to control conversion.

`mdl-toolkit inference` uses a similar input format for inference. All options of `mdl-toolkit convert-dataset` (except input/output) also apply to `mdl-toolkit inference` and should be kept consistent between training and inference.

**General options**

* `--model-name`: default `mispeech/midashenglm-7b`. Optional for convert-dataset and training. Hugging Face model name or local path.
* `--from-modelscope`: default `false`. Whether to load the model from ModelScope instead of Hugging Face. Requires the `modelscope` extra; see Installation at installation.md.
* `--tokenizing-batch-size`: default `8`. Batch size used for tokenization.
* `--num-workers`: default (dynamic). Number of worker processes for data processing. By default, half of available CPU cores, capped at 32. Due to implementation details, this only parallelizes part of the preprocessing pipeline.

**Dataset options**

* `--system-prompt`: default `null`. Default system prompt to guide model behavior. If the dataset has a `system_prompt` column, its non-null values override this default.
* `--user-prompt`: default `null`. Default user prompt. If the dataset has a `user_prompt` column, its non-null values override this default.
* `--base-dir`: default `null`. Base directory for resolving relative paths in the dataset. If not set, paths are resolved relative to the current working directory.

**Input and output**

* `INPUT`: required positional. Path to the input CSV dataset.
* `--output`: required. Path to write the processed dataset. Existing files will be overwritten.

## `mdl-toolkit train` — fine-tune a model with a dataset

The `mdl-toolkit train` command fine-tunes a pretrained model on the given dataset and saves the resulting model. If an evaluation dataset is configured, it runs evaluation during training and reports validation loss. Checkpoints are saved automatically by default for recovery.

It accepts either a CSV dataset or a preconverted dataset. If a CSV is provided, it will be converted first, and all options of `mdl-toolkit convert-dataset` (except input/output) also apply. If a preconverted dataset is provided, conversion options are ignored.

**Training options**

* `--train-dataset`: required. Path to the training dataset.
* `--lr`: default `1e-4`. Learning rate.
* `--lora-rank`: default `32`. LoRA rank. Higher rank captures more features but increases compute/storage and overfitting risk. For simple tasks, try 8–16; for complex tasks, 32 or higher, usually not exceeding 128.
* `--lora-alpha`: default `32`. LoRA alpha scaling.
* `--lora-dropout`: default `0`. LoRA dropout rate.
* `--train-target`: default `["encoder", "projector", "decoder"]`. Target modules to train. Choose from `encoder`, `projector`, `decoder`, `embed_tokens`, `lm_head`. Can be specified multiple times. If `embed_tokens` and `lm_head` are chosen, they will be fully trained.
* `--num-epochs`: default `1`. Total epochs. For LLMs, 1–3 epochs are often enough. Larger values rarely help and may overfit. Fractions are allowed for partial-epoch training.
* `--warmup-steps`: default `0`. Warmup steps.

**Memory options**

* `--batch-size`: default `8`. Per-device batch size. If gradient accumulation or multi-GPU is used, effective batch size is `batch_size * gradient_accumulation_steps * num_gpus`. LLM fine-tuning is usually insensitive to batch size; choose based on memory.
* `--gradient-accumulation-steps`: default `1`. Steps to accumulate gradients before an optimizer step.
* `--gradient-checkpointing`: default `true`. Enable gradient checkpointing to save memory at the cost of extra compute.
* `--bf16`: default (dynamic). Use bfloat16 if supported; reduces memory and may speed up compute with slight precision trade-offs.
* `--quantization`: default `null`. Quantize model weights (`8bit` or `4bit`). Reduces memory with some compute overhead and potential minor quality impact. Requires the `quantization` extra; see installation.md.

**Evaluation options**

* `--eval-dataset`: optional. Path to the evaluation dataset. If omitted, no evaluation is run and other eval options are ignored.
* `--eval-steps`: default `500`. Evaluate every N steps.
* `--eval-batch-size`: default `null`. Per-device eval batch size. If unset, falls back to training batch size. Because eval is forward-only, a larger batch is often possible.
* `--eval-accumulation-steps`: default `null`. Accumulate eval results across steps to reduce transfer overhead.
* `--report-to`: default `[]`, repeatable. Report metrics to the specified platforms. See transformers docs for supported values.

**Checkpointing and output**

* `--output`: required. Output directory. Checkpoints and final artifacts are written here.
* `--resume-from-checkpoint`: default `null`. Resume training from a checkpoint. `null` or `false` starts fresh. `true` resumes from the last checkpoint. A path resumes from that specific checkpoint.
* `--save-steps`: default `500`. Save a checkpoint every N steps (int >= 1) or every fraction of an epoch for values in [0, 1).
* `--save-total-limit`: default `null`. Max number of checkpoints to keep. If set, the oldest are removed when exceeding the limit.
* `--merge-lora`: default `true`. Merge LoRA adapters into the base model before exporting. Produces a stand-alone model at the cost of extra disk space. If disabled, only the LoRA adapters and modified weights are saved.

## `mdl-toolkit inference` — run inference with a model

The `mdl-toolkit inference` command provides a simple interface to run the model on given inputs and produce outputs. Use the same system and user prompts as used during training to ensure the output format matches expectations.

It targets quick post-training checks and is not optimized for performance or flexibility. For production, consider `vllm` or other specialized inference frameworks.

It accepts the same input schema as training. All options of `mdl-toolkit convert-dataset` (except input/output) also apply and should remain consistent between training and inference.

**Inference options**

* `INPUT`: required positional. Path to the input CSV dataset.
* `--output`: required. Path to the output CSV. Existing files will be overwritten.
* `--model-name`: required. HF model name or local path for inference.
* `--batch-size`: default `32`. Per-device batch size for inference.
* `--max-length`: default `128`. Maximum sequence length including input, output, and special tokens. Outputs beyond this length are truncated; inputs longer than this cause an error.
