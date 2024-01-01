import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d, InstanceNorm1d

class BinaryClassification(nn.Module):
  def __init__(self, embed_size, device):
    super(BinaryClassification, self).__init__()
    # Number of input features is embed_size.
    self.layer_1 = Linear(embed_size, 64)
    self.layer_2 = Linear(64, 64)
    self.layer_out = Linear(64, 1)

    self.relu = ReLU()
    self.dropout = Dropout(p=0.5)
    self.batchnorm1 = BatchNorm1d(64)
    # self.instance_norm1 = InstanceNorm1d(64)
    self.batchnorm2 = BatchNorm1d(64)
    # self.instance_norm2 = InstanceNorm1d(64)
    self.device = device

  def forward(self, inputs):
    # x = inputs.view(inputs.size(0), -1)

    x = self.relu(self.layer_1(inputs))

    # if inputs.size(0) > 1:
    #   x = self.batchnorm1(x)
    # else:
    #   x = self.instance_norm1(x)

    x = self.batchnorm1(x)
    x = self.relu(self.layer_2(x))

    # if inputs.size(0) > 1:
    #   x = self.batchnorm2(x)
    # else:
    #   x = self.instance_norm2(x)

    x = self.batchnorm2(x)
    x = self.dropout(x)
    x = self.layer_out(x)

    return x