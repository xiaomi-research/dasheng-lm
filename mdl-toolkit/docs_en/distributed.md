# Distributed Training

MDL-Toolkit supports distributed training via `torchrun` and `accelerate`. To use distributed training, prepend the appropriate launcher to your training command.

## Single-node training with `torchrun`

Use the following command to utilize all GPUs on one node.

Arguments:
* `--standalone`: Run in standalone mode; `torchrun` autoconfigures the rendezvous backend locally.
* `--nproc-per-node gpu`: Number of processes per node. `gpu` uses all available GPUs.
* `--no-python`: Run the subsequent command directly without going through the Python interpreter.

```bash
torchrun --standalone --nproc-per-node gpu --no-python \
    mdl-toolkit train \
    --lora-rank 16 \
    --eval-steps 50 \
    --train-dataset train-converted/ \
    --eval-dataset test-converted/ \
    --output output/
```

## Multi-node training with `torchrun`

Ensure all nodes can reach each other over the network, then run the following on each node.

Arguments:
* `--nnodes $NUM_NODES`: Number of nodes. Replace `$NUM_NODES` accordingly.
* `--rdzv-backend c10d`: Rendezvous backend, typically `c10d`.
* `--rdzv-endpoint $HOST_NODE_ADDR`: Rendezvous endpoint as `<host>[:<port>]`. Must be consistent across all nodes.
* `--rdzv-id $JOB_ID`: Unique job ID. Replace `$JOB_ID` accordingly.

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

## Distributed training with `accelerate`

Install the `accelerate` package and run the following to configure the environment interactively:

```bash
accelerate config
```

This guides you through choosing the distributed type, number of nodes/GPUs, etc. Defaults are fine unless you have specific needs.

To use multiple config files (e.g., one per rank on a shared filesystem), specify the config path explicitly:

```bash
accelerate config --config_file /path/to/config/file
```

For single-node multi-GPU, choose "MULTI_GPU", set number of machines to 1, and pick the GPU count. Example config for 8 GPUs:

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

For multi-node, choose "MULTI_GPU" with `num_machines > 1`, set the main node IP and the current node's rank. Example for 2 nodes with 8 GPUs each:

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
