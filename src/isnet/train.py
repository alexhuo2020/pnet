import torch 
from tqdm import tqdm
import time 
def train(models, eq, train_dataloaders, eval_dataloaders, optimizers, lr_schedulers, train_config, device = "cpu"):
    train_loss = []
    val_loss = []
    if train_config.save_metrics:
        train_step_loss = []
        eval_step_loss = []
        metrics_filename = f"{train_config.output_dir}/metrics_data-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        total_length = len(train_dataloaders[0])
        pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
        total_loss = 0.0
        for step, batch in enumerate(zip(train_dataloaders[0], train_dataloaders[1])):
            models.unet.train()
            batch = {"x": batch[0], "xb":batch[1]}
            for key in batch.keys():
                batch[key] = batch[key].to(device)
                batch[key].requires_grad = True
            loss = models.loss_u(batch, eq)
            if train_config.save_metrics:
                train_step_loss.append(loss.detach().float().item())
            loss.backward()
            optimizers.optim_u.step()
            optimizers.optim_u.zero_grad()
            total_loss += loss.detach().float()

            models.vnet.train()
            loss_v = models.loss_v(batch, eq)
            loss_v.backward()
            optimizers.optim_v.step()
            optimizers.optim_v.zero_grad()
            
            pbar.update(1)
            err = eq.compute_err(models.unet,batch['x'])
            pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloaders[0])} completed (loss: {loss.detach().float()}, error: {err.detach().float()})")
            if train_config.save_metrics:
                save_to_json(metrics_filename, train_step_loss, train_loss)
        pbar.close()

    epoch_end_time = time.perf_counter() - epoch_start_time
    epoch_times.append(epoch_end_time)
    train_epoch_loss = total_loss / len(train_dataloaders[0])
    train_loss.append(float(train_epoch_loss))

    if lr_schedulers is not None:
        lr_schedulers.scheduler_u.step()
        lr_schedulers.scheduler_v.step()
    
    


def save_to_json(output_filename, train_step_loss, train_epoch_loss, val_step_loss, val_epoch_loss):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
