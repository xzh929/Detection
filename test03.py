import torch

gain = torch.tensor([ 1.,  1., 80., 80., 80., 80.,  1.])
print(gain[[2,3]])