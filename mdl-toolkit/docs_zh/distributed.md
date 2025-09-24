# 分布式训练

MDL-Toolkit 支持使用`torchrun`和`accelerate`进行分布式训练。要使用分布式训练，只需在训练命令前添加相应的启动命令。

## 使用`torchrun`在单个节点上进行分布式训练

使用以下命令利用单个节点上的全部 GPU 进行训练。

参数说明：
* `--standalone`：以独立模式运行，`torchrun`将自动配置本地会合后端。
* `--nproc-per-node gpu`：指定每个节点上运行的进程数量。指定为`gpu`将使用所有可用的 GPU。
* `--no-python`：直接运行后续的命令，而不需要通过 Python 解释器。

```bash
torchrun --standalone --nproc-per-node gpu --no-python \
    mdl-toolkit train \
    --lora-rank 16 \
    --eval-steps 50 \
    --train-dataset train-converted/ \
    --eval-dataset test-converted/ \
    --output output/
```

## 使用`torchrun`在多个节点上进行分布式训练

要在多个节点上进行分布式训练，需要确保所有节点能够通过网络互相访问，并在每个节点上运行以下命令。

参数说明：
* `--nnodes $NUM_NODES`：指定参与训练的节点数量。应将`$NUM_NODES`替换为实际的节点数量。
* `--rdzv-backend c10d`：指定会合后端。通常使用`c10d`。
* `--rdzv-endpoint $HOST_NODE_ADDR`：指定会合后端的地址。应将`$HOST_NODE_ADDR`替换为`<host>[:<port>]`格式的地址。地址可以是任意节点的地址，但必须确保该地址在所有节点上保持一致。
* `--rdzv-id $JOB_ID`：指定训练作业的唯一 ID。应将`$JOB_ID`替换为一个唯一的作业 ID。

```bash
torchrun \
    --nnodes $NUM_NODES \
    --nproc-per-node gpu \
    --rdzv-backend c10d \
    --rdzv-endpoint $HOST_NODE_ADDR \
    --rdzv-id $JOB_ID \
    --no-python \
    mdl-toolkit train \
    --lora-rank 16 \
    --eval-steps 50 \
    --train-dataset train-converted/ \
    --eval-dataset test-converted/ \
    --output output/
```

## 使用`accelerate`进行分布式训练

要使用`accelerate`进行分布式训练，请确保已安装`accelerate`库。运行以下命令以交互式方式配置分布式训练环境：

```bash
accelerate config
```

这将引导你完成配置过程，其中关键选项包括选择分布式类型、指定节点和 GPU 数量等。对于其他选项，除非有特殊需求，否则可以使用默认值。

如果希望使用多个不同的配置文件，例如，在共享文件系统上创建不同 Rank 的配置，可以使用以下命令指定配置文件的路径：

```bash
accelerate config --config_file /path/to/config/file
```

要在单个节点上进行分布式训练，需要在配置时选择“多 GPU”选项，设置节点数量为 1，并指定使用的 GPU 数量。使用 8 个 GPU 的示例配置文件内容如下：

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

要在多个节点上进行分布式训练，需要在配置时选择“多节点”选项，并指定主节点的地址和当前节点的 Rank。建议在单个节点上创建初始配置文件，将配置文件分发到所有节点，然后修改`machine_rank`以匹配每个节点的 Rank。使用 2 个节点、每个节点 8 个 GPU 的示例配置文件内容如下：

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_process_ip: 10.0.0.1
main_process_port: 29500
main_training_function: main
mixed_precision: 'no'
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
