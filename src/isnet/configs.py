from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1#00
    vocab_size: int = 5 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    # n_embd: int = 128
    dropout: float = 0.
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    out_d: int=1

class eq_config:
    name: int = "Poisson"
    d: int = 5

class model_config:
    names = "MLP"
    hdim: int = 200
    depth: int = 4
    d: int = 5

class data_config:
    name = 'box'
    num_int: int = 100000
    num_ext: int = 1000
    batch_size: int = 1000
    d: int = 5
    box_low: float = -1.0
    box_high: float = 1.0


@dataclass
class train_config:
    device: str='cuda'
    model_name: str="elliptic"
    enable_fsdp: bool=True
    enable_ddp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=False
    batch_size: int=1000
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=10000
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.2 #0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=False
    # val_batch_size: int=1
    dataset = "box"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    lr_step_size: int = 100#5000


# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD # HYBRID_SHARD "Full Shard within a node DDP cross Nodes", SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP".
    hsdp : bool =False # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    sharding_group_size : int=0 # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int=0 #requires hsdp to be set. This specifies the replica group size, which is world_size/sharding_group_size.
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool=True
    fsdp_cpu_offload: bool=False
    pure_bf16: bool = False
    optimizer: str= "AdamW"
