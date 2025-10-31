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

from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliSubCommand,
)

from .convert_dataset import ConvertDatasetCli
from .inference import InferenceCli
from .train import TrainCli


class Cli(
    BaseSettings,
    cli_parse_args=True,
    cli_kebab_case=True,
    cli_enforce_required=True,
):
    train: CliSubCommand[TrainCli]
    convert_dataset: CliSubCommand[ConvertDatasetCli]
    inference: CliSubCommand[InferenceCli]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(Cli)
