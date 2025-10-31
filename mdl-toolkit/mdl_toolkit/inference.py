# Copyright 2025 Horizon Team, MiLM Plus, Xiaomi Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
from pathlib import Path

from pydantic_settings import CliPositionalArg
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .convert_dataset import ConvertConfig, padding, process_data, transpose


class InferenceConfig(ConvertConfig):
    model_name: str
    batch_size: int = 32
    max_length: int = 128


class InferenceCli(InferenceConfig):
    input: CliPositionalArg[Path]
    output: Path
    base_dir: Path | None = None

    def cli_cmd(self) -> None:
        inference(self)


def inference(config: InferenceCli) -> None:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
        device_map="auto",
    )

    ds = process_data(config=config, input_path=config.input, mode="generation")
    ds = ds.batch(config.batch_size, num_proc=config.num_workers)
    with (
        open(config.input, "r") as in_file,
        open(config.output, "w") as out_file,
    ):
        reader = csv.DictReader(in_file)
        assert reader.fieldnames is not None, "Input CSV must have headers"
        fields = reader.fieldnames
        if "prediction" not in fields:
            fields = [*fields, "prediction"]
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()

        reader_iter = iter(reader)
        for batch in tqdm(
            ds,
            desc="Inference",
            dynamic_ncols=True,
        ):
            batch = padding(
                list(transpose(batch)),  # type: ignore[arg-type]
                tokenizer=tokenizer,
                dtype=model.dtype,
                device=model.device,
            )
            outputs = model.generate(
                **batch,
                max_length=config.max_length,
                return_dict_in_generate=False,
            )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prediction in predictions:
                row = next(reader_iter)
                row["prediction"] = prediction
                writer.writerow(row)
