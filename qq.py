import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

import matplotlib.pyplot as plt
from flax import linen as nn 
from flax.training import train_state, checkpoints 

class MLP(nn.Module):
    hdim: int 
    depth: int 

    def setup(self):
        self.linear = [nn.Dense(self.hdim) for _ in range(self.depth)]
        # for _ in range(depth):
        #     self.linear.append(nn.Dense(hdim))
        #     self.linear.append(nn.tanh)
        self.out = nn.Dense(1)
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.linear):
            x = lyr(x)
            if i != len(self.linear) - 1:
                x = nn.relu(x)
        return self.out(x)
main_rng = random.PRNGKey(42)

# main_rng, x_rng = random.split(main_rng)
# x = random.normal(x_rng, (3,2))
# model = MLP(hdim=10, depth=2)
# params = model.init(x_rng, x)
# y = model.apply(params,x)      
# print(y.shape)  
# define the loss of each neural network
# params_other are the parameters of i-1 neural networks
def loss_i(params,params_other,x):
  yi = batched_grad(params,x)
  zi = batched_predict(params,x)
  y = [batched_grad(params_other[i],x) for i in range(len(params_other))]
  z = [batched_predict(params_other[i],x) for i in range(len(params_other))]
  y.append(yi)
  z.append(zi)
  y, z = y_to_v(y,z)
  return jnp.mean(jnp.sum(y[-1]**2,1))/jnp.mean(z[-1]**2)
  
@jax.jit
def loss(state, batch):
    state.apply_fn({'params': state.params}, batch)
    loss = 
