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
