import torch

def load_data(d, num_int, num_ext, box= [-1,1], num_workers=0):
  """
  Mento carlo sampling on -1--1 
  :param d: dimension
  :param num_int: number of interior sampling points
  :param num_ext: number of exterior sampling points
  :param batch_size: batch size
  Return:
    two data loader, first for the interior points, second for the exterior points
  """

  x_dist = torch.distributions.Uniform(box[0], box[1])
  xs = x_dist.sample((num_int,d))
  xb = x_dist.sample((2*d,num_ext,d))
  for dd in range(d):
    xb[dd,:,dd] = torch.ones(num_ext)*box[1]
    xb[dd + d,:,dd] =  torch.ones(num_ext)*box[0]
  xb = xb.reshape(2*d*num_ext,d)
  return xs, xb 

def load_L_datand(d, num_int, num_ext):
    base_dist0 = torch.distributions.uniform.Uniform(-1,1)
    base_dist1 = torch.distributions.uniform.Uniform(0,1)
    x = base_dist0.sample((2*num_int,d))
    ind = torch.prod(x > 0,-1)==0
    x = x[ind,:]
    # if len(ind) < num_int:
    #    x = x[ind,:]
    # else:
    #    x = x[:num_int,:]
    
    xb = base_dist0.sample((2*d,2*num_ext,d))
    for dd in range(d):
        xb[dd,:,dd] = torch.ones(2*num_ext)
        xb[dd + d,:,dd] =  -torch.ones(2*num_ext)
    xb = xb.reshape(2*d*2*num_ext,d)
    ind = torch.prod(xb>0,-1)==0
    xb = xb[ind,:]

    xb1 = base_dist1.sample((2*d,num_ext,d))
    for dd in range(d):
        xb1[dd,:,dd] = torch.ones(num_ext)
        xb1[dd + d,:,dd] =  torch.zeros(num_ext)
    xb1 = xb1.reshape(2*d*num_ext,d)
    ind = torch.prod(xb1>0,-1)==0
    xb1 = xb1[ind,:]
    xb = torch.concat([xb,xb1])
    idx =  torch.randperm(x.shape[0])
    x = x[idx].view(x.size())
    idx = torch.randperm(xb.shape[0])
    xb = xb[idx].view(xb.size())
    return x, xb




def load_L_data2d(num_int, num_ext):
    """
    Mento Carlo sampling on the L-shape data 
    (-1,-1)->(1,-1)->(1,0)->(0,0)->(0,1)->(-1,1)->(-1,-1)
    Return:
      two dataset, first for interior points, second for boundary data points
    """
    d = 2
    base_dist1 = torch.distributions.uniform.Uniform(0,1)
    base_dist0 = torch.distributions.uniform.Uniform(-1,0)
    x = base_dist0.sample((int(num_int/3),d))
    xx1 = torch.cat([base_dist0.sample((int(num_int/3),1)),base_dist1.sample((int(num_int/3),1))],-1)
    xx2 = torch.cat([base_dist1.sample((int(num_int/3),1)),base_dist0.sample((int(num_int/3),1))],-1)
    x = torch.concat([x,xx1,xx2])
    bsize=num_ext
    x1 = torch.cat([base_dist0.sample((bsize,1)),torch.ones(bsize,1)],-1)
    x2 = torch.cat([base_dist0.sample((bsize,1)),-torch.ones(bsize,1)],-1)
    x3 = torch.cat([-torch.ones(bsize,1),base_dist0.sample((bsize,1))],-1)
    x4 = torch.cat([-torch.ones(bsize,1),base_dist1.sample((bsize,1))],-1)
    x5 = torch.cat([torch.ones(bsize,1),base_dist0.sample((bsize,1))],-1)
    x6 = torch.cat([torch.zeros(bsize,1),base_dist1.sample((bsize,1))],-1)
    x7 = torch.cat([base_dist1.sample((bsize,1)),-torch.ones(bsize,1)],-1)
    x8 = torch.cat([base_dist1.sample((bsize,1)),torch.zeros(bsize,1)],-1)
    xb = torch.concat([x1,x2,x3,x4,x5,x6,x7,x8])
    return x, xb


def load_dataset(data_config, split = "train"):
    if data_config.name == "box":
        x, xb = load_data(data_config.d, data_config.num_int, data_config.num_ext, box= [data_config.box_low,data_config.box_high])
    elif data_config.name == "Lshape":
        x, xb = load_L_datand(data_config.d, data_config.num_int, data_config.num_ext)
    else:
        print("dataset not provided by default package")
        raise NotImplementedError
    if split == "train":
        x = x[int(len(x)/20):]
        xb = xb[int(len(xb)/20):]
    elif split == "test":
        x = x[:int(len(x)/20)]
        xb = xb[:int(len(xb)/20)]
    else:
        x = x
        xb = xb
        # print("specify train or test, no other values are allowed")
        # raise NotImplementedError
    return x, xb

   
if __name__ == "__main__":
    class a:
        pass
    data_config = a()
    data_config.name = "box"
    data_config.d = 2
    data_config.num_int = 10000
    data_config.num_ext = 100
    data_config.box_low = 0
    data_config.box_high = 1
    x, xb = load_dataset(data_config,"")
    print(x.shape, xb.shape)
    data_config.name = "Lshape"
    x, xb = load_dataset(data_config)
    print(x.shape, xb.shape)
