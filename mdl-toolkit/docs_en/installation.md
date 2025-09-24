# MDL-Toolkit Installation

It is recommended to install `mdl-toolkit` into a dedicated virtual environment to avoid dependency conflicts. You can create virtual environments with `uv`, `conda`/`mamba`, or `venv`, or run `mdl-toolkit` in an isolated environment via `uvx` or `pipx`.

## Optional features

`mdl-toolkit` ships with some optional features that require extra dependencies:

- `modelscope`: Integrates ModelScope model hub to load and use pretrained models from ModelScope.
- `quantization`: Supports loading quantized models and quantizing non-quantized models to reduce GPU memory usage during fine-tuning.

To install these options, use the `[extras]` syntax, e.g., `mdl-toolkit[modelscope,quantization]`.

## Run with `uvx`

You can run `mdl-toolkit` in an isolated environment using `uvx`:

```bash
uvx mdl-toolkit --help
# Or, with optional features
uvx --from 'mdl-toolkit[modelscope,quantization]' mdl-toolkit --help
```

## Create a virtual environment and install

Create a virtual environment with `uv`, `venv`, or `conda`/`mamba`, then install `mdl-toolkit`:

```bash
# Using uv
uv venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# Or, using venv
python -m venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# Or, using conda/mamba
mamba create -n mdl-toolkit python=3.13 pip
mamba activate mdl-toolkit

# Install mdl-toolkit
pip install mdl-toolkit
# Or, with optional features
pip install 'mdl-toolkit[modelscope,quantization]'

# Now you can use mdl-toolkit
mdl-toolkit --help
```

## Install from source

You can install the latest development version of `mdl-toolkit` from a Git repository using a VCS URL:

```bash
# Using uvx
uvx --from 'mdl-toolkit @ git+https://github.com/xiaomi-research/dasheng-lm.git#subdirectory=mdl-toolkit' mdl-toolkit --help

# Or, create and activate a virtual environment first
uv venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate
# Then install with pip
pip install 'mdl-toolkit @ git+https://github.com/xiaomi-research/dasheng-lm.git#subdirectory=mdl-toolkit'
```

You can also install from a locally cloned repository or extracted source archive:

```bash
# Clone the repo
git clone https://github.com/xiaomi-research/dasheng-lm.git
# Or download and extract the source archive

# Create and activate a virtual environment
uv venv path/to/mdl-toolkit-venv
source path/to/mdl-toolkit-venv/bin/activate

# Install mdl-toolkit
pip install 'mdl-toolkit @ ./dasheng-lm/mdl-toolkit'
```
