import torch.nn as nn
import torch.nn.functional as F
import torch

from .prune import PruningModule, MaskedLinear

class ElementTensors:
    def __init__(self, model):
        self.layer_tensors = {}
        weights_in = 0
        weights_out = torch.zeros((1,1)).cuda()
        
        for name, param in model.named_parameters():
            if len(param.size()) > 1 and 'weight' in name:
                layer_name = name.split('.')[0]
                weights_in = weights_out
                weights_out = param
                
                if weights_in.size(0) == weights_out.size(1):
                    weights_group = torch.cat((weights_in, weights_out.t()), dim=1)
                    self.layer_tensors[layer_name] = torch.ones_like(
                    weights_group, requires_grad=False)
                # else:
                #     self.layer_tensors[layer_name] = torch.ones_like(
                #     weights_out, requires_grad=False)
                    # self.layer_tensors[layer_name + '2'] = torch.ones_like(
                    # weights_in, requires_grad=False)
                
                
                
                
class StructureTensors:
    def __init__(self, model):
        self.layer_tensors = {}

        for name, param in model.named_parameters():
            if len(param.size()) > 1 and 'weight' in name:

                layer_name = name.split('.')[0]
                self.layer_tensors[layer_name] = torch.ones(
                param.size(1), requires_grad=False)
                
                
                
                
class EltwiseLayer(nn.Module):
  def __init__(self, n, train):
    super(EltwiseLayer, self).__init__()
    self.weights = nn.Parameter(torch.ones([1, n]), requires_grad=train)  # define the trainable parameter

  def forward(self, x):
    # assuming x is of size b-1-h-w
    return x * self.weights  # element-wise multiplication

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)


    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    
class LeNet_act(PruningModule):
    def __init__(self, train=True):
        super(LeNet_act, self).__init__()
        linear = nn.Linear
        self.act1 = EltwiseLayer(784,train)
        self.fc1 = linear(784, 300)
        self.act2 = EltwiseLayer(300,train)
        self.fc2 = linear(300, 100)
        self.act3 = EltwiseLayer(100,train)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(self.act1(x)))
        x = F.relu(self.fc2(self.act2(x)))
        x = F.log_softmax(self.fc3(self.act3(x)), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        #self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(800, 500)
        self.fc2 = linear(500, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        #x = self.conv3(x)
        #x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
