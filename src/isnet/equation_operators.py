import torch 
from torch.autograd import grad
def opA(u,x):
    d = len(x[0])
    #Compute Laplacian
    u_x = grad(u, x,
                    create_graph=True, retain_graph=True,
                    grad_outputs=torch.ones_like(u),
                    allow_unused=True)[0]
    u_xx = 0
    for i in range(d):
        u_xx += grad(u_x[:,i], x, retain_graph=True,
                        create_graph=True,
                        grad_outputs=torch.ones_like(u_x[:,i]),
                        allow_unused=True)[0][:,i]
    return u_xx

def opB(u,x):
   return u.squeeze()

def opA_semilinear(u,x,a):
    d = len(x[0])
    #Compute Laplacian
    u_x = grad(u, x,
                    create_graph=True, retain_graph=True,
                    grad_outputs=torch.ones_like(u),
                    allow_unused=True)[0]
    u_xx = 0
    for i in range(d):
        u_xx += grad(a(x)*u_x[:,i], x, retain_graph=True,
                        create_graph=True,
                        grad_outputs=torch.ones_like(u_x[:,i]),
                        allow_unused=True)[0][:,i]
    return u_xx

def opA_nonlinear(u,x,a):
    d = len(x[0])
    #Compute Laplacian
    u_x = grad(u, x,
                    create_graph=True, retain_graph=True,
                    grad_outputs=torch.ones_like(u),
                    allow_unused=True)[0]
    u_xx = 0
    for i in range(d):
        u_xx += grad(a(u)*u_x[:,i], x, retain_graph=True,
                        create_graph=True,
                        grad_outputs=torch.ones_like(u_x[:,i]),
                        allow_unused=True)[0][:,i]
    return u_xx
def opB_nonlinear(u,x):
    u_x = grad(u, x,
                create_graph=True, retain_graph=True,
                grad_outputs=torch.ones_like(u),
                allow_unused=True)[0]
    n = torch.zeros_like(x)
    for i in range(len(x[0])):
        n[(x==0)[:,i],i]=-1
        n[(x==1)[:,i],i]=1
    n[(x==0)[:,0],:]=0
    n[(x==1)[:,0],:]=0
    return u[:,0]*(x==0)[:,0] + (u[:,0]-1)*(x==1)[:,0] + torch.sum(n*u_x,dim=-1)

def opB_mixed(u,x):
    n = torch.zeros_like(x)
    n[(x==0)[:,0],0] = 0.
    n[(x==0)[:,1],1] = -1.
    n[(x==1)[:,0],0] = 0.
    n[(x==1)[:,1],1] = 1.
    u_x = grad(u, x,
            create_graph=True, retain_graph=True,
            grad_outputs=torch.ones_like(u),
            allow_unused=True)[0]
    return torch.sum(u_x * n,dim=-1)  + u[:,0] * ((x==0)[:,0] + (x==1)[:,0])
