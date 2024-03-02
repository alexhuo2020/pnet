import torch
import torch.nn as nn
from isnet.configs import model_config
import math 
# torch.set_default_dtype(torch.float64)
class MODEL(nn.Module):
  """
  The MLP for representing solutions of PDEs and the Lagrangian multiplier
  :param d: dimensions of the domain
  :param hdim: number of neurons in hidden layers
  :param depth: number of hidden layers
  :param act: activation function, default tanh
  """
  def __init__(self, d=2, hdim=200, depth=1, act=nn.Tanh(), out_d = 1):
    super().__init__()
    self.d = d
    self.hdim = hdim
    self.depth = depth
    self.act = act
    layers = nn.ModuleList()
    layers.append(nn.Linear(d,hdim))
    layers.append(act)
    for i in range(depth):
      layers.append(nn.Linear(hdim,hdim))
      layers.append(act)
    layers.append(nn.Linear(hdim,out_d))
    self.l = nn.Sequential(*layers)

  def forward(self, x):
    xx = self.l(x)
    return xx

  def get_num_params(self):
    """
    Return the number of parameters in the model.
    """
    n_params = sum(p.numel() for p in self.parameters())
    return n_params
 
  def configure_optimizers(self, train_config):
    # optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.lr, betas=train_config.betas, weight_decay=train_config.weight_decay)
    optimizer = torch.optim.RMSprop(self.parameters(), lr=train_config.lr)
    return optimizer

  def from_pretrained(self,path):
    self.load_state_dict(torch.load(path))


def load_model(model_config, device="cpu"):
    unet = MODEL(d = model_config.d, hdim = model_config.hdim, depth = model_config.depth, out_d = 1)
    vnet = MODEL(d = model_config.d, hdim = model_config.hdim, depth = model_config.depth, out_d = 1)
    return (unet.to(device), vnet.to(device))

class MODELS:
  def __init__(self, nets):
    self.unet, self.vnet = nets
  def configure_optimizers(self, train_config):
    self.optim_u = self.unet.configure_optimizers(train_config) 
    self.optim_v = self.vnet.configure_optimizers(train_config) 
  def configure_schedulers(self, train_config):
    self.scheduler_u = None
    self.scheduler_v = None 
  def loss_u(self, ds, eq):
    u = self.unet(ds['x'])
    u_xx = eq.opA(u,ds['x'])
    v = self.vnet(ds['x']).squeeze()
    # Boundary
    ubB = eq.opB(self.unet(ds['xb']), ds['xb'])
    loss_u = (0.5*torch.mean((ubB-eq.g(ds['xb']))**2) + torch.mean((- u_xx - eq.f(ds['x']))*v.detach()))
    return loss_u
  def loss_v(self, ds, eq):
    u = self.unet(ds['x'])
    u_xx = eq.opA(u,ds['x'])
    v = self.vnet(ds['x']).squeeze()
    loss_v =  torch.mean((u_xx.detach() + eq.f(ds['x']))*v)
    return loss_v 
