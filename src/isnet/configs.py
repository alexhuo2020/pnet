from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 2 # dimension of the problem
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    out_d: int=1


class eq_config:
    name: int = "Poisson"
    d: int = 2

class model_config:
    names = "GPT"
    hdim: int = 50
    depth: int = 2
    d: int = 2

class data_config:
    name = 'box'
    num_int: int = 10000
    num_ext: int = 100
    batch_size: int = 1000
    d: int = 2
    box_low: float = -1.0
    box_high: float = 1.0


@dataclass
class train_config:
    model_name: str="elliptic"
    enable_fsdp: bool=False
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
    lr: float=1e-3
    weight_decay: float=0.0
    gamma: float= 0.2 #0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
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