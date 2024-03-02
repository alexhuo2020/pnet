# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import torch 
from isnet.configs import model_config, data_config, train_config, eq_config
from isnet.models import MODELS, load_model
from isnet.equations import load_equation
from isnet.models_transformer import Model_GPT
from isnet.models_gpt import GPT, GPTConfig
from transformers import GPT2Config 
from isnet.dataset import load_dataset

from time import time 
import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from isnet.train import train


def main(**kwargs):
    # class a:
    #     pass
    # models = a()
    train_config.device='cuda'
    print(model_config.names)
    if model_config.names == "MLP":
        models = MODELS(load_model(model_config, train_config.device))
        models.configure_optimizers(train_config)
    elif model_config.names == "GPT":
        # configs = GPT2Config(vocab_size=2, n_positions=1, n_embd=32, n_layer = 4, n_head = 2)
        # unet = Model_GPT(configs)
        # print(unet)
        # vnet = Model_GPT(configs)
        unet = GPT(GPTConfig).to(train_config.device)
        vnet = GPT(GPTConfig).to(train_config.device)
        models = MODELS((unet,vnet))
        
        # models.configure_optimizers(train_config)
        models.optimizers = (unet.configure_optimizers(train_config), vnet.configure_optimizers(train_config))
        models.optim_u, models.optim_v = models.optimizers
        models.optim_u = torch.optim.RMSprop(unet.parameters(), lr=train_config.lr)
        models.optim_v = torch.optim.RMSprop(vnet.parameters(), lr=train_config.lr)

    # print_model_size(models.unet, train_config, device='cpu')
     
    dataset_x_train, dataset_xb_train =  load_dataset(data_config, split="train")
    dataset_x_eval, dataset_xb_eval =  load_dataset(data_config, split="test")


    # # Create DataLoaders for the training and validation dataset
    train_dataloader_x = torch.utils.data.DataLoader(
        dataset_x_train,
        batch_size = train_config.batch_size,
    )
    train_dataloader_xb = torch.utils.data.DataLoader(
        dataset_xb_train,
        batch_size = len(dataset_xb_train)// int(len(dataset_x_train)//train_config.batch_size),
    )
    eval_dataloader_x = torch.utils.data.DataLoader(
        dataset_x_eval,
        batch_size = 1,
    )
    eval_dataloader_xb = torch.utils.data.DataLoader(
        dataset_xb_eval,
        batch_size = 1,
    )


    class op():
        pass 
    optimizers = op()
    optimizers.optim_u, optimizers.optim_v = models.optim_u, models.optim_v


    schedulers = op()
    schedulers.scheduler_u = StepLR(optimizers.optim_u, step_size=train_config.lr_step_size, gamma=train_config.gamma) # 
    schedulers.scheduler_v = StepLR(optimizers.optim_v, step_size=train_config.lr_step_size, gamma=train_config.gamma) # 

    train_dataloaders = train_dataloader_x, train_dataloader_xb
    eval_dataloaders = eval_dataloader_x, eval_dataloader_xb

    eq = load_equation(eq_config)
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
        train_config,
    )

if __name__ == "__main__":
    # fire.Fire(main)
    main()