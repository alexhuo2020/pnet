"""define the equation"""
import torch
from torch.autograd import grad
from functools import partial
import isnet.equation_operators as op
# from equation_operators import opA, opB, opA_nonlinear, opB_nonlinear, opA_semilinear, opB_mixed
class Poisson:
  """
  the class of infsupnet
  :param d: dimension of domain
  :param f: source function in the equation
  :param g: boundary data
  :param A: the elliptic operator
  :param B: the operator for the boundary condition

  """
  def __init__(self, d, f, g, opA, opB, ur=None):
    self.d = d
    self.f = f
    self.g = g
    self.opA = opA
    self.opB = opB
    self.ur = ur
  def compute_err(self,uf,x):
    return (torch.linalg.norm(uf(x) - self.ur(x))) / (torch.linalg.norm(self.ur(x)))


def load_equation(eq_config):
    if eq_config.name == "Poisson":
        f = lambda x: eq_config.d*torch.pi**2/4*torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range(eq_config.d)]),0)
        g = lambda x: 0
        ur = lambda x: torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range(eq_config.d)]),0).unsqueeze(1)
        eq = Poisson(eq_config.d, f, g, op.opA, op.opB, ur)
        return eq 
    elif eq_config.name == "semilinear":
        f = lambda x:  -(2*eq_config.d+2)*torch.sum(x,dim=-1)
        ur = lambda x: 1 + torch.sum(x**2,dim=-1) 
        g = ur
        eq = Poisson(eq_config.d, f, g, op.opA_semilinear, op.opB, ur)
        return eq 
    elif eq_config.name == "nonlinear":
        f = lambda x: 0.
        m = 3
        ur = lambda x: ((2**(m+1)-1)*x[:,0]+1)**(1/(m+1)) - 1.
        g = lambda x: 0
        a_x = lambda x: (1 + x[:,0])**m
        opA = partial(op.opA_nonlinear,a=a_x)
        eq = Poisson(eq_config.d, f, g, op.opA, op.opB_nonlinear, ur)
        return eq
    elif eq_config.name == "mixedBC":
        g = lambda x: torch.sin(5*x[:,0])*((x==1)[:,1] + (x==0)[:,1])
        def f(x):
            return 10*torch.exp(-(((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2))/0.02)
        eq = Poisson(eq_config.d, f, g, op.opA, op.opB_mixed, ur)
        return eq
    else:
       print("equation not implemented!")
       raise NotImplementedError



       
    

if __name__=="__main__":
   from configs import eq_config
   print(load_equation(eq_config))