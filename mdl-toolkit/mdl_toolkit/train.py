import os
from functools import partial
from pathlib import Path
from typing import Literal, cast

import torch
from accelerate import PartialState  # type: ignore[import-untyped]
from datasets import Dataset, load_from_disk  # type: ignore[import-untyped]
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing_extensions import assert_never

from .convert_dataset import ConvertConfig, padding, process_data


class TrainConfig(ConvertConfig):
    lr: float = 1e-4
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0
    train_target: set[
        Literal["encoder", "projector", "decoder", "embed_tokens", "lm_head"]
    ] = {
        "encoder",
        "projector",
        "decoder",
    }
    num_epochs: float = 1.0
    warmup_steps: int = 0

    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    bf16: bool | None = None
    quantization: Literal["8bit", "4bit"] | None = None

    eval_steps: int | float = 500
    eval_batch_size: int | None = None
    eval_accumulation_steps: int | None = None
    report_to: list[str] = []

    save_steps: int | float = 500
    save_total_limit: int | None = None
    merge_lora: bool = True


class TrainCli(TrainConfig):
    train_dataset: Path
    eval_dataset: Path | None = None
    resume_from_checkpoint: Path | bool | None = None
    output: Path

    def cli_cmd(self) -> None:
        train(self)


def load_dataset(config: ConvertConfig, path: str) -> Dataset:
    if path.endswith(".csv"):
        return process_data(config=config, input_path=path, mode="train")
    else:
        return cast(Dataset, load_from_disk(path))


def train(config: TrainCli) -> None:
    state = PartialState()
    print(f"Distributed: {state.distributed_type}")
    if state.distributed_type != "NO":
        print(f"Rank: {state.process_index} (local: {state.local_process_index})")

    model_dtype = (
        torch.bfloat16
        if config.bf16 is True
        or (config.bf16 is None and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    if config.from_modelscope:
        from modelscope import snapshot_download  # type: ignore[import-untyped]

        model_name = snapshot_download(config.model_name)
    else:
        model_name = config.model_name

    processor: ProcessorMixin = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    train_ds = load_dataset(config, os.fspath(config.train_dataset))
    eval_ds = (
        load_dataset(config, os.fspath(config.eval_dataset))
        if config.eval_dataset is not None
        else None
    )

    quantization_config: BitsAndBytesConfig | None
    match config.quantization:
        case "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_dtype,
            )
        case "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        case None:
            quantization_config = None
        case _:
            assert_never(config.quantization)

    model: PreTrainedModel | PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=model_dtype,
        device_map="auto",
        **(
            dict(quantization_config=quantization_config)
            if quantization_config is not None
            else {}
        ),
    )

    print(f"Model loaded with {model.dtype}")

    target_modules = []
    modules_to_save = []
    if "encoder" in config.train_target:
        target_modules.append(r"^audio_encoder\.blocks\.\d+\.attn\.(qkv|proj)$")
    if "projector" in config.train_target:
        target_modules.append(r"^audio_projector\.net\.(0|2)$")
    if "decoder" in config.train_target:
        target_modules.append(
            r"^decoder\.model\.layers\.\d+\.(self_attn|mlp)\.(up|gate|down)_proj$"
        )
    if "embed_tokens" in config.train_target:
        modules_to_save.append("embed_tokens")
    if "lm_head" in config.train_target:
        modules_to_save.append("lm_head")

    model = cast(
        PeftModel,
        get_peft_model(
            cast(PreTrainedModel, model),
            LoraConfig(
                r=config.lora_rank,
                target_modules="|".join(target_modules),
                exclude_modules=["lm_head"],
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                modules_to_save=modules_to_save,
                task_type="CAUSAL_LM",
            ),
        ),
    )
    model.print_trainable_parameters()

    output_dir = os.fspath(config.output)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size or config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        learning_rate=config.lr,
        num_train_epochs=config.num_epochs,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=config.eval_steps,
        logging_steps=1,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        warmup_steps=config.warmup_steps,
        report_to=config.report_to,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=partial(
            padding,
            tokenizer=tokenizer,
            dtype=model_dtype,
            device=model.device,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    if torch.cuda.is_available():
        print(
            f"Peak VRAM during loading: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GiB"
        )
        torch.cuda.reset_peak_memory_stats()

    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    if state.is_main_process:
        print(result)

    if torch.cuda.is_available():
        print(
            f"Peak VRAM during training: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GiB"
        )

    if config.merge_lora:
        model = model.merge_and_unload()

    if state.is_main_process:
        final_path = os.fspath(config.output / "final")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        processor.save_pretrained(final_path)
