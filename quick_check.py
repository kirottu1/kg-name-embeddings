
import torch
from siamese_nn import SiameseNetwork

net = SiameseNetwork()  # 128 + 768
x1 = torch.randn(4, 128, dtype=torch.float64)
x2 = torch.randn(4, 768, dtype=torch.float64)
y  = net(x1, x2)  # [4, 1]
print(y.shape, y.dtype)  # torch.Size([4, 1]) torch.float64
