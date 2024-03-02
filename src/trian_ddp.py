# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import torch 
from isnet.configs import model_config, data_config, train_config, eq_config
from isnet.models import MODELS, load_model
from isnet.equations import load_equation
from isnet.models_gpt import GPT 
from isnet.configs import GPTConfig
from isnet.dataset import load_dataset


import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from isnet.configs import train_config, fsdp_config

from isnet.train_multi import (
    train,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from accelerate.utils import is_xpu_available

def main(**kwargs):
    # Update the configuration for the training and sharding process
    # train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    # fsdp_config = FSDP_CONFIG()
    # update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    else:
        torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or train_config.enable_ddp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        else:
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            if model_config.names == "MLP":
                models = MODELS(load_model(model_config))
                models.configure_optimizers(train_config)
            elif model_config.names == "GPT":
                models.unet = GPT(GPTConfig)
                models.vnet = GPT(GPTConfig)
                models.optimizers = (models.unet.configure_optimizers(train_config), models.vnet.configure_optimizers(train_config))
    else:
        if model_config.names == "MLP":
            models = MODELS(load_model(model_config))
            models.configure_optimizers(train_config)
        elif model_config.names == "GPT":
            models.unet = GPT(GPTConfig)
            models.vnet = GPT(GPTConfig)
            models.optimizers = (models.unet.configure_optimizers(train_config), models.vnet.configure_optimizers(train_config))

    print_model_size(models.unet, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    # if train_config.quantization: # NOT IMPLEMENTED YET
    #     models = prepare_model_for_int8_training(models)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        models.unet.to(torch.bfloat16)
        models.vnet.to(torch.bfloat16)

    # if train_config.use_peft: # NOT IMPLEMENTED
    #     peft_config = generate_peft_config(train_config, kwargs)
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        # if not train_config.use_peft and train_config.freeze_layers: # NOT IMPLEMENTED

            # freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = None #fsdp_auto_wrap_policy(models.unet, LlamaDecoderLayer) #NOT IMPLEMENTED

        fsdp_config.fsdp_cpu_offload = False
        models.unet = FSDP(
            models.unet,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.xpu.current_device() if is_xpu_available() else torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        # if fsdp_config.fsdp_activation_checkpointing:
        #     apply_fsdp_checkpointing(models.unet)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            models.unet.to("xpu:0")
            models.vnet.to("xpu:0")
        else:
            models.unet.to("cuda")
            models.vnet.to("cuda")
    
    dataset_x_train, dataset_xb_train =  load_dataset(data_config, split="train")
    dataset_x_eval, dataset_xb_eval =  load_dataset(data_config, split="test")


    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_x_train)}")

    # dataset_val = get_preprocessed_dataset(
    #     tokenizer,
    #     dataset_config,
    #     split="test",
    # )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_x_eval)}")

    # if train_config.batching_strategy == "packing":
    #     dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    # train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
    # batch_size = 1000
    # numbatch = 100

    # # Create DataLoaders for the training and validation dataset
    train_dataloader_x = torch.utils.data.DataLoader(
        dataset_x_train,
        batch_size = train_config.batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        # **train_dl_kwargs
    )
    train_dataloader_xb = torch.utils.data.DataLoader(
        dataset_xb_train,
        batch_size = len(dataset_xb_train)// int(len(dataset_x_train)//train_config.batch_size),
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        # **train_dl_kwargs
    )
    eval_dataloader_x = torch.utils.data.DataLoader(
        dataset_x_eval,
        batch_size = 1,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        # **train_dl_kwargs
    )
    eval_dataloader_xb = torch.utils.data.DataLoader(
        dataset_xb_eval,
        batch_size = 1,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        # **train_dl_kwargs
    )


    #     )
    class op():
        pass 
    optimizers = op()
    optimizers.optim_u, optimizers.optim_v = models.optim_u, models.optim_v

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizers.optim_u = AnyPrecisionAdamW(
            models.unet.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
        optimizers.optim_v = AnyPrecisionAdamW(
            models.vnet.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        pass
        # optimizers.optim_u = optim.AdamW(
        #     models.unet.parameters(),
        #     lr=train_config.lr,
        #     weight_decay=train_config.weight_decay,
        # )
        # optimizers.optim_v = optim.AdamW(
        #     models.vnet.parameters(),
        #     lr=train_config.lr,
        #     weight_decay=train_config.weight_decay,
        # )
    schedulers = op()
    schedulers.scheduler_u = StepLR(optimizers.optim_u, step_size=train_config.lr_step_size, gamma=train_config.gamma) # 
    schedulers.scheduler_v = StepLR(optimizers.optim_v, step_size=train_config.lr_step_size, gamma=train_config.gamma) # 

    train_dataloaders = train_dataloader_x, train_dataloader_xb
    eval_dataloaders = eval_dataloader_x, eval_dataloader_xb

    eq = load_equation(eq_config)#Poisson(d=2, f=lambda x:0, g=lambda x:0, nu=1, ur=None)
    train_config.save_metrics=False
    train_config.gradient_clipping=False
    train_config.run_validation = False
    # Start the training process
    results = train(
        models,
        eq,
        train_dataloaders,
        eval_dataloaders,
        optimizers,
        schedulers,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)