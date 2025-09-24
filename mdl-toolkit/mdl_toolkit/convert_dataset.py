import multiprocessing
import os
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import torch
from datasets import Dataset  # type: ignore[import-untyped]
from pydantic import Field
from pydantic_settings import CliPositionalArg
from transformers import AutoProcessor, AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .conversation import DatasetConfig, build_conversation


class ConvertConfig(DatasetConfig):
    model_name: str = "mispeech/midashenglm-7b"
    from_modelscope: bool = False
    tokenizing_batch_size: int = 8
    num_workers: int = Field(
        default_factory=lambda: max(1, min(32, multiprocessing.cpu_count() // 2)),
    )


def transpose(batch: dict[str, list[str]]) -> Iterable[dict[str, str]]:
    assert len(batch) > 0
    num_rows = len(next(iter(batch.values())))
    assert all(len(v) == num_rows for v in batch.values()), (
        "All columns must have the same length"
    )

    for i in range(num_rows):
        yield {key: value[i] for key, value in batch.items()}


def process_data(
    config: ConvertConfig,
    input_path: str | os.PathLike,
    mode: Literal["train", "generation"],
) -> Dataset:
    if config.from_modelscope:
        from modelscope import snapshot_download  # type: ignore[import-untyped]

        model_name = snapshot_download(config.model_name)
    else:
        model_name = config.model_name

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    # Avoid pickling issues
    get_processor = cache(
        lambda: cast(
            ProcessorMixin,
            AutoProcessor.from_pretrained(model_name, trust_remote_code=True),
        )
    )

    def apply_chat_template(batch: dict[str, list[str]]) -> dict[str, torch.Tensor]:
        return get_processor().apply_chat_template(
            conversation=list(
                build_conversation(
                    row,
                    config,
                    with_prediction=mode == "train",
                )
                for row in transpose(batch)
            ),
            tokenize=True,
            add_special_tokens=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=mode == "generation",
        )

    start_of_user = tokenizer.encode("<|im_start|>user\n")
    start_of_assistant = tokenizer.encode("<|im_start|>assistant\n")

    def derive_labels(example):
        input_ids = cast(list[int], example["input_ids"])

        def find_all_subsequences(seq: list[int], subseq: list[int]) -> list[int]:
            indexes = []
            for i in range(len(seq) - len(subseq) + 1):
                if seq[i : i + len(subseq)] == subseq:
                    indexes.append(i)
            return indexes

        user_starts = find_all_subsequences(input_ids, start_of_user)
        assistant_starts = find_all_subsequences(input_ids, start_of_assistant)

        retained_range = []
        while True:
            if not assistant_starts:
                break
            while user_starts and user_starts[0] < assistant_starts[0]:
                user_starts.pop(0)
            retained_range.append(
                slice(
                    assistant_starts.pop(0),
                    user_starts.pop(0) if user_starts else None,
                )
            )

        labels = [-100] * len(input_ids)
        for r in retained_range:
            labels[r] = input_ids[r]

        return {"labels": labels}

    dataset = Dataset.from_csv(os.fspath(input_path))
    dataset = dataset.map(
        apply_chat_template,
        # Result of apply_chat_template is always batched, so we set batched=True
        # even if batching is not strictly necessary
        batched=True,
        batch_size=config.tokenizing_batch_size,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )
    if mode == "train":
        dataset = dataset.map(
            derive_labels,
            num_proc=config.num_workers,
            desc="Deriving labels for training",
        )

    return dataset


class _MDLModelInput(TypedDict, total=False):
    input_ids: list[int]
    attention_mask: list[int]
    input_values: list[float]
    labels: list[int]


def padding(
    batch: list[_MDLModelInput],
    *,
    tokenizer: PreTrainedTokenizerBase,
    dtype: torch.dtype,
    device: str | torch.device | int | None = None,
) -> dict[str, Any]:
    assert len(batch) > 0, "Batch must not be empty"

    max_text_length = max(len(example["input_ids"]) for example in batch)
    max_audio_length = max(len(example["input_values"]) for example in batch)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    result: list[_MDLModelInput] = []
    for example in batch:
        assert len(example["input_ids"]) == len(example["attention_mask"])
        assert "labels" not in example or len(example["labels"]) == len(
            example["input_ids"]
        )

        num_text_padding = max_text_length - len(example["input_ids"])
        num_audio_padding = max_audio_length - len(example["input_values"])
        result.append(
            {
                "input_ids": [pad_token_id] * num_text_padding
                + example.pop("input_ids"),
                "attention_mask": [0] * num_text_padding
                + example.pop("attention_mask"),
                "input_values": example.pop("input_values") + [0.0] * num_audio_padding,
                **(
                    {"labels": [-100] * num_text_padding + example.pop("labels")}
                    if "labels" in example
                    else {}
                ),
                **example,
            }
        )

    tensors: dict[str, torch.Tensor] = {}
    for key in result[0].keys():
        values = [example[key] for example in result]  # type: ignore[literal-required]
        tensor = torch.tensor(values, device=device)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)
        tensors[key] = tensor
    return tensors


class ConvertDatasetCli(ConvertConfig):
    input: CliPositionalArg[Path]
    output: Path

    def cli_cmd(self) -> None:
        dataset = process_data(config=self, input_path=self.input, mode="train")
        if len(dataset) == 0:
            raise ValueError(
                "Processed dataset is empty. Please check your input data."
            )
        dataset.save_to_disk(self.output)
