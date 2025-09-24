# MDL-Toolkit

English | [中文](README_zh.md)

MDL-Toolkit is a user-friendly MiDashengLM fine-tuning toolkit that wraps the entire MDL fine-tuning workflow into a unified CLI. It uses a simple CSV data format and a LoRA-based approach to provide out-of-the-box fine-tuning, supports various memory optimization options and distributed training, works across GPU clusters of all sizes, and offers a quick inference command to help you efficiently complete fine-tuning tasks.

## Installation

It is strongly recommended to install `mdl-toolkit` into a dedicated virtual environment to avoid dependency conflicts with other projects.

To install `mdl-toolkit`, you can use the following commands:

```bash
# Create and activate a dedicated virtual environment with uv
uv venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# Or, use venv
python -m venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# Or, use conda/mamba
mamba create -n mdl-toolkit python=3.13 pip
mamba activate mdl-toolkit

# Install mdl-toolkit
pip install mdl-toolkit
# Or, if you need optional features
pip install 'mdl-toolkit[modelscope,quantization]'

# You can now use the mdl-toolkit command
mdl-toolkit --help
```

For more installation options, please refer to the [Installation Guide](docs_en/installation.md).

## Usage

This section describes how to use `mdl-toolkit` for model training. We also provide a Jupyter Notebook demonstrating [fine-tuning MiDashengLM with ESC-50](docs_en/esc-50.ipynb).

### Data Preparation

Before starting training, you need to prepare the dataset. `mdl-toolkit` uses a CSV-formatted dataset, where each row represents one audio sample, and the first row must contain column names. Irrelevant columns will be ignored. The dataset can contain the following columns:

- `audio`: **Required**. The path to the audio file, or a URL starting with `http://` or `https://`. The specified path will be resolved relative to the directory where the script is run or the base directory specified by the `--base-dir` option. The specified URL will be downloaded when generating the dataset.
- `system_prompt`: *Optional*. System prompt text. If not provided or is `null`, the command-line option will be used if provided; otherwise it will be set to empty.
- `user_prompt`: *Optional*. User prompt text. If not provided or is `null`, the command-line option will be used if provided; otherwise it will be set to empty.
- `prediction`: **Required** for training; the model's predicted output, which will be used as labels for supervised learning during training. For inference, this column will be ignored and replaced with the inference result.

For example, for the ESC-50 dataset, you can use the following format:

```csv
audio,prediction
audio/1-100032-A-0.wav,"target: 0, category: dog"
audio/1-100038-A-14.wav,"target: 14, category: chirping_birds"
audio/1-100210-A-36.wav,"target: 36, category: vacuum_cleaner"
```

You can optionally specify system and user prompts:

```csv
audio,system_prompt,user_prompt,prediction
audio/1-100032-A-0.wav,null,What is the sound in the audio?,It sounds like a dog barking.
audio/1-100038-A-14.wav,Classify the audio according to the ESC-50 categories.,null,chirping_birds
audio/1-100210-A-36.wav,Answer user's question about the audio.,Is that a vacuum cleaner?,Yes.
```

System and user prompts can also be specified using command-line options.

### Converting the Dataset

Running `mdl-toolkit convert-dataset` converts the CSV-formatted dataset into the format required for model training. The command reads the input CSV, loads audio files, performs necessary preprocessing, and saves the results to the specified output directory. Converting the dataset is optional—you can directly pass the CSV file to the training command to process it on the fly—but preconverting allows reuse across multiple training runs and improves efficiency.

```bash
mdl-toolkit convert-dataset \
    path/to/input.csv \
    --output path/to/output/
```

### Training

Use the `mdl-toolkit train` command to start training. This command reads the converted dataset, loads the base model, and trains using default hyperparameters.

```bash
mdl-toolkit train \
    --train-dataset path/to/converted/train/ \
    --eval-dataset path/to/converted/eval/ \
    --output path/to/output/
```

If you don't use an evaluation set, you can omit the `--eval-dataset` parameter.

During training, logs such as loss values and learning rate will be printed. Checkpoints will be saved under `checkpoint-{step}` subdirectories of the output directory. Training may take a long time depending on the dataset size, model size, and hardware. After training completes, the results will be saved under the `final` subdirectory of the output directory. By default, the `final` directory contains the full model weights with LoRA adapters merged, and you can load and use this model the same way as the base model.

#### Tuning Hyperparameters

`mdl-toolkit` provides a set of tunable hyperparameters to help optimize model performance during training. You can specify these hyperparameters via command-line options, for example:

```bash
mdl-toolkit train \
    --lr 1e-4 \
    --lora-rank 32 \
    ...
```

`mdl-toolkit` provides default values for all hyperparameters, but the defaults may not be suitable for all tasks. Below are some commonly used hyperparameters and their default values:

- `--lr`: **Default: `1e-4`**. Learning rate, controls the rate at which the optimizer updates parameters.
- `--lora-rank`: **Default: `32`**. The rank of LoRA, which controls the complexity of the LoRA adapters. A higher rank can capture more features but also increases compute and storage overhead and the risk of overfitting.
- `--batch-size`: **Default: `8`**. The number of samples processed per GPU device in each training step. A larger batch size may improve training speed and stability but also increases memory usage.

For the full list of hyperparameters, default values, and other available options, please refer to the [Command-Line Interface Reference](docs_en/cli.md).

#### Distributed Training

`mdl-toolkit` is compatible with `torchrun` or `accelerate`. To use distributed training, simply prepend the corresponding launcher. If you don't use distributed training, it will run on a single GPU by default. For more information, refer to the [Distributed Training Guide](docs_en/distributed.md).

For example, to use `torchrun` for distributed training on a single node:

```bash
torchrun --standalone --nproc-per-node gpu --no-python \
    mdl-toolkit train \
    --train-dataset path/to/converted/train/ \
    --eval-dataset path/to/converted/eval/ \
    --output path/to/output/
```

To use `torchrun` for multi-node distributed training, run the same command on each node, ensure all nodes can reach each other over the network, replace `$NUM_NODES` with the actual number of nodes, `$JOB_ID` with a unique job ID, and `$HOST_NODE_ADDR` with the address (and optional port) of the host node in the form `<host>[:<port>]`:

```bash
torchrun --nnodes $NUM_NODES --nproc-per-node gpu \
    --rdzv-id $JOB_ID \
    --rdzv-backend c10d \
    --rdzv-endpoint $HOST_NODE_ADDR \
    --no-python \
    mdl-toolkit train \
    --train-dataset path/to/converted/train/ \
    --eval-dataset path/to/converted/eval/ \
    --output path/to/output/
```

To use `accelerate` for distributed training, first run `accelerate config` on each node for configuration, then launch training with `accelerate launch`:

```bash
accelerate config # Follow the interactive prompts
accelerate launch \
    mdl-toolkit train \
    --train-dataset path/to/converted/train/ \
    --eval-dataset path/to/converted/eval/ \
    --output path/to/output/
```

### Inference

To run inference with the merged model, the usage is the same as the base model. Some frameworks support loading LoRA adapters directly for inference. During inference, the system and user prompts fed to the model should match those used during training to ensure the model outputs as expected:

```python
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Load the merged model from the final training output
model_path = "path/to/output/final/"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "System prompt"}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "User prompt"},
            {"type": "audio", "path": "/path/to/example.wav"},
        ],
    },
]
```

For large-scale inference, it's recommended to use the [vLLM](../README.md#deploy-with-vllm) framework for better performance and more comprehensive features.

In addition, MDL-Toolkit provides an inference command based on `transformers`, which makes it convenient to quickly run basic inference tasks after training, though it doesn't perform as well as specialized inference frameworks like vLLM. The inference input is a CSV file with the same format as the training dataset, except the `prediction` column becomes optional. The inference output will copy all input columns and replace the `prediction` column with the model's predictions. You can run inference with the following command:

```bash
mdl-toolkit inference \
    path/to/input.csv \
    --output path/to/output.csv \
    --model-name model_name_or_path \
    --max-length 128
```
