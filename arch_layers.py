import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 10, 5)
    self.conv2 = nn.Conv2d(10, 20, 3)
    self.fc1 = nn.Linear(20*10*10, 500)
    self.fc2 = nn.Linear(500, 5)
  def forward(self,x):
    in_size = x.size(0)
    out = self.conv1(x)
    out = F.relu(out)
    out = F.max_pool2d(out, 2, 2)
    out = self.conv2(out)
    out = F.relu(out)
    out = out.view(in_size, -1)
    out = self.fc1(out)
    out = F.relu(out)
    out = self.fc2(out)
    out = F.log_softmax(out, dim=1)
    return out

# class ConvNet(nn.Module):
#   def __init__(self, in_shape, out_dim):
#     super(ConvNet, self).__init__()
#     self.conv1 = nn.Sequential(
#       nn.Conv2d(in_shape[0], 16, 3, 1, 1),
#       nn.BatchNorm2d(16),
#       nn.MaxPool2d(2),
#       nn.ReLU(),
#     )
#     self.conv2 = nn.Sequential(
#       nn.Conv2d(16, 32, 3, 1, 1),
# 			nn.BatchNorm2d(32),
# 			nn.MaxPool2d(2),
# 			nn.ReLU()
#     )
#     self.conv3 = nn.Sequential(
#       nn.Conv2d(32, 64, 3, 1, 1),
# 			nn.BatchNorm2d(64),
# 			nn.MaxPool2d(2),
# 			nn.ReLU()
#     )
#     self.conv4 = nn.Sequential(
#       nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU()
#     )
#     self.conv5 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.BatchNorm2d(64),
# 		  nn.ReLU()
# 		)
#     self.mlp1 = nn.Linear(10816, 125)
#     # self.mlp1 = nn.Sequential(
# 		# 	nn.Linear(10816, 125),
# 		#   nn.ReLU()
#     # )
#     self.mlp2 = nn.Linear(125, out_dim)

#   def forward(self, x):
#     x1 = self.conv1(x)
#     x2 = self.conv2(x1)
#     x3 = self.conv3(x2)
#     x4 = self.conv4(x3)
#     x5 = self.conv5(x4)
#     out1 = self.mlp1(x5.view(x5.size(0), -1))
#     out2 = self.mlp2(out1)
#     return out2