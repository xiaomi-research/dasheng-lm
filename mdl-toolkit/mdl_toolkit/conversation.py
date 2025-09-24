import os
from pathlib import Path
from typing import NewType

from pydantic import BaseModel

Conversation = NewType("Conversation", list)


class DataRow(BaseModel):
    audio: str
    system_prompt: str | None = None
    user_prompt: str | None = None
    prediction: str | None = None


class DatasetConfig(BaseModel):
    system_prompt: str | None = None
    user_prompt: str | None = None
    base_dir: Path | None = None


def build_conversation(
    row: dict[str, str],
    config: DatasetConfig,
    with_prediction: bool,
) -> Conversation:
    row_ = DataRow.model_validate(row)

    audio = os.path.join(config.base_dir, row_.audio) if config.base_dir else row_.audio
    system_prompt = row_.system_prompt or config.system_prompt
    user_prompt = row_.user_prompt or config.user_prompt
    prediction = row_.prediction
    if with_prediction:
        assert prediction is not None, "`prediction` is required"
    else:
        prediction = None

    return Conversation(
        [
            *(
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                ]
                if system_prompt
                else []
            ),
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    *([{"type": "text", "text": user_prompt}] if user_prompt else []),
                ],
            },
            *(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": prediction}],
                    }
                ]
                if with_prediction
                else []
            ),
        ]
    )
