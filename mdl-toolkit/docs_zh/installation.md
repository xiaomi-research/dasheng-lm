# MDL-Toolkit 安装

建议将`mdl-toolkit`安装到专用的虚拟环境中，以避免与其他项目的依赖冲突。可以使用`uv`、`conda`/`mamba`、`venv`等工具创建虚拟环境，或使用`uvx`、`pipx`等工具在隔离环境中安装和运行`mdl-toolkit`。

## 可选功能

`mdl-toolkit`提供了一些可选功能，这些功能需要额外安装依赖包。所有可选功能的列表如下：

- `modelscope`：集成 ModelScope 模型库，支持加载和使用 ModelScope 中的预训练模型。
- `quantization`：支持加载量化模型和对未量化模型进行量化，以减少微调时的显存占用。

要安装这些可选功能，可以使用`[extras]`语法，例如：`mdl-toolkit[modelscope,quantization]`。

## 使用`uvx`运行

可以使用`uvx`在隔离环境中运行`mdl-toolkit`：

```bash
uvx mdl-toolkit --help
# 或者，如果需要可选功能
uvx --from 'mdl-toolkit[modelscope,quantization]' mdl-toolkit --help
```

## 创建虚拟环境并安装

可以使用`uv`、`venv`、`conda`/`mamba`等工具创建虚拟环境并安装`mdl-toolkit`：

```bash
# 使用 uv
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

## 从源代码安装

可以使用 VCS URL 从 Git 仓库安装最新开发版的`mdl-toolkit`：

```bash
# 使用 uvx
uvx --from 'mdl-toolkit @ git+https://github.com/xiaomi-research/dasheng-lm.git#subdirectory=mdl-toolkit' mdl-toolkit --help

# 或者，使用任意方式创建并激活虚拟环境
uv venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# 然后使用 pip 安装
pip install 'mdl-toolkit @ git+https://github.com/xiaomi-research/dasheng-lm.git#subdirectory=mdl-toolkit'
```

也可以从克隆的本地仓库或下载的源代码安装`mdl-toolkit`：

```bash
# 克隆仓库
git clone https://github.com/xiaomi-research/dasheng-lm.git
# 或者，下载源代码并解压

# 使用任意方式创建并激活虚拟环境
uv venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate

# 安装 mdl-toolkit
pip install 'mdl-toolkit @ ./dasheng-lm/mdl-toolkit'
```
