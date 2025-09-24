# MDL-Toolkit

[English](README.md) | 中文

MDL-Toolkit 是用户友好的 MiDashengLM 微调工具包，将 MDL 微调全流程封装为统一的 CLI 界面，采用简洁的 CSV 数据格式，基于 LoRA 方案，提供开箱即用的微调功能，支持各种显存优化选项和分布式训练，适用于各种规模的 GPU 集群，并提供快捷的推理命令，助力用户高效完成微调任务。

## 安装

强烈建议将`mdl-toolkit`安装到专用的虚拟环境中，以避免与其他项目的依赖冲突。

要安装`mdl-toolkit`，可以使用以下命令：

```bash
# 使用 uv 创建并激活专用虚拟环境
uv venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# 或者，使用 venv
python -m venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# 或者，使用 conda/mamba
mamba create -n mdl-toolkit python=3.13 pip
mamba activate mdl-toolkit

# 安装 mdl-toolkit
pip install mdl-toolkit
# 或者，如果需要可选功能
pip install 'mdl-toolkit[modelscope,quantization]'

# 现在可以使用 mdl-toolkit 命令
mdl-toolkit --help
```

有关更多安装选项，请参考[安装指南](docs_zh/installation.md)。

## 用法

本节介绍如何使用`mdl-toolkit`进行模型训练。我们还提供了一个 Jupyter Notebook，演示[使用 ESC-50 对 MiDashengLM 进行微调](docs_zh/esc-50.ipynb)。

### 数据准备

在开始训练之前，需要准备好数据集。`mdl-toolkit`使用 CSV 格式的数据集，每行代表一个音频样本，其中第一行必须包含列名。无关的列将被忽略。数据集可以包含以下列：

- `audio`：**必需**，音频文件的路径，或以`http://`或`https://`开头的 URL 。指定的路径将相对于运行脚本的目录或`--base-dir`选项指定的基目录解析，指定的 URL 将在生成数据集时下载音频文件。
- `system_prompt`：*可选*，系统提示文本。如果未提供或为`null`，将尝试使用命令行选项，如果未提供命令行选项，将设置为空。
- `user_prompt`：*可选*，用户提示文本。如果未提供或为`null`，将尝试使用命令行选项，如果未提供命令行选项，将设置为空。
- `prediction`：对于训练**必需**，模型的预测输出，训练时将其作为标签进行监督学习。对于推理将被忽略，并使用推理结果替换。

例如，对于 ESC-50 数据集，可以使用以下格式：

```csv
audio,prediction
audio/1-100032-A-0.wav,"target: 0, category: dog"
audio/1-100038-A-14.wav,"target: 14, category: chirping_birds"
audio/1-100210-A-36.wav,"target: 36, category: vacuum_cleaner"
```

可以选择性地指定系统提示和用户提示：

```csv
audio,system_prompt,user_prompt,prediction
audio/1-100032-A-0.wav,null,What is the sound in the audio?,It sounds like a dog barking.
audio/1-100038-A-14.wav,Classify the audio according to the ESC-50 categories.,null,chirping_birds
audio/1-100210-A-36.wav,Answer user's question about the audio.,Is that a vacuum cleaner?,Yes.
```

系统提示和用户提示也可以使用命令行选项指定。

### 转换数据集

运行`mdl-toolkit convert-dataset`会将 CSV 格式的数据集转换为模型训练所需的格式。该命令会读取输入 CSV 文件、加载音频文件、完成必要的预处理并将结果保存到指定的输出目录中。转换数据集是可选的，可以在训练时直接指定 CSV 文件以在训练过程中进行处理，但预先转换数据集可以在多次训练间复用转换结果，提高训练效率。

```bash
mdl-toolkit convert-dataset \
    path/to/input.csv \
    --output path/to/output/
```

### 训练

使用`mdl-toolkit train`命令启动模型训练。该命令会读取转换后的数据集，加载基础模型，并使用默认超参数进行训练。

```bash
mdl-toolkit train \
    --train-dataset path/to/converted/train/ \
    --eval-dataset path/to/converted/eval/ \
    --output path/to/output/
```

如果不使用评估集，可以省略`--eval-dataset`参数。

训练时会输出训练过程中的日志信息，包括损失值、学习率等，并在输出目录的`checkpoint-{step}`子目录中保存检查点。训练可能需要较长时间，具体取决于数据集大小、模型大小和硬件配置。训练完成后，会在输出目录的`final`子目录中保存训练结果。默认情况下，`final`目录中包含已合并 LoRA 适配器的完整模型权重，可以使用与基础模型相同的方式加载和使用该模型。

#### 调整超参数

`mdl-toolkit`为用户提供了一组可调节的超参数，以便在训练过程中优化模型性能。可以通过命令行选项指定这些超参数，例如：

```bash
mdl-toolkit train \
    --lr 1e-4 \
    --lora-rank 32 \
    ...
```

`mdl-toolkit`为所有超参数提供了默认值，但默认值不一定适用于所有任务。以下是一些常用超参数及其默认值：

* `--lr`：**默认值：`1e-4`** 学习率，控制优化器更新参数的速率。
* `--lora-rank`：**默认值：`32`** LoRA 的秩，控制 LoRA 适配器的复杂度。较高的秩可以捕捉更多的特征，但也会增加计算和存储开销，并增加过拟合的风险。
* `--batch-size`：**默认值：`8`** 每个训练步骤中每个 GPU 设备处理的样本数量。较大的批量大小可能会提高训练速度并增加模型的稳定性，但也会增加内存使用量。

完整的超参数列表、默认值和其他可用选项请参考[命令行界面参考](docs_zh/cli.md)。

#### 分布式训练

`mdl-toolkit`兼容`torchrun`或`accelerate`。要使用分布式训练，只需添加相应的启动命令。如果不使用分布式训练，则默认在单个 GPU 上运行。有关更多信息，请参考[分布式训练指南](docs_zh/distributed.md)。

例如，要使用`torchrun`在单一节点上进行分布式训练：

```bash
torchrun --standalone --nproc-per-node gpu --no-python \
    mdl-toolkit train \
    --train-dataset path/to/converted/train/ \
    --eval-dataset path/to/converted/eval/ \
    --output path/to/output/
```

要使用`torchrun`多个节点上进行分布式训练，需要在每个节点上运行相同的命令，确保所有节点能够通过网络互相访问，并将`$NUM_NODES`替换为实际的节点数量，将`$JOB_ID`替换为唯一的作业 ID，将`$HOST_NODE_ADDR`替换为主节点的地址加上可选的端口号，格式为`<host>[:<port>]`：

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

要使用`accelerate`进行分布式训练，需要首先在每个节点上运行`accelerate config`进行配置，随后可以使用`accelerate launch`命令启动训练：

```bash
accelerate config # 根据提示完成交互式配置
accelerate launch \
    mdl-toolkit train \
    --train-dataset path/to/converted/train/ \
    --eval-dataset path/to/converted/eval/ \
    --output path/to/output/
```

### 推理

使用合并后的模型进行推理，其推理方式与基础模型相同。部分框架支持直接加载 LoRA 适配器进行推理。推理时，输入模型的系统提示和用户提示应与训练时保持一致，以确保模型输出的内容符合预期：

```python
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# 从最终训练输出加载合并后的模型
model_path = "path/to/output/final/"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "系统提示文本"}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "用户提示文本"},
            {"type": "audio", "path": "/path/to/example.wav"},
        ],
    },
]
```

对于大规模推理，推荐使用[vLLM](../README.md#deploy-with-vllm)框架以获得更好的性能和更全面的功能支持。

此外，MDL-Toolkit 基于`transformers`提供了一个推理命令，便于用户在训练后快速运行基本的推理任务，但性能上不如vLLM等专用推理框架。推理输入为 CSV 文件，其格式与训练数据集相同，除了`prediction`列变为可选内容。推理输出将复制输入数据的所有列，并将`prediction`列替换为模型的预测结果。可以使用以下命令运行推理：

```bash
mdl-toolkit inference \
    path/to/input.csv \
    --output path/to/output.csv \
    --model-name model_name_or_path \
    --max-length 128
```
