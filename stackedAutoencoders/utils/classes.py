import torch.nn as nn

import torch.nn.functional as F

import torch



class TensorflowConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation_fn=None):
        super(TensorflowConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.zeros(out_channels))  # Learnable bias
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.conv(x) + self.bias.view(1, -1, 1, 1)  # Convolutional operation with bias
        if self.activation_fn:
            out = self.activation_fn(out)  # Apply activation function
        return out
    
    def set_weights(self, new_weights, device):
        if new_weights[0] is not None:
            self.conv.weight.data = torch.from_numpy(new_weights[0]).to(device)
        if new_weights[1] is not None:
            self.bias.data = torch.from_numpy(new_weights[1]).to(device)

class TensorflowLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=None):
        super(TensorflowLinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))  # Separate bias
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.linear(x) + self.bias  # Linear operation with separate bias
        if self.activation_fn:
            out = self.activation_fn(out)  # Apply activation function
        return out
    
    def set_weights(self, new_weights, device):
        if new_weights[0] is not None:
            self.linear.weight.data = torch.from_numpy(new_weights[0]).to(device)
        if new_weights[1] is not None:
            self.bias.data = torch.from_numpy(new_weights[1]).to(device)
    
class InputLayer(nn.Module):
    def __init__(self):
        super(InputLayer, self).__init__()

    def forward(self, x):
        return x
    
class TensorflowMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(TensorflowMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.ind = None
        self.M = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, return_indices=True)
    
    def forward(self, x):
        out, self.ind = self.M(x)
        return out

class BasicConvModel(nn.Module):
    def __init__(self, inc, outc, ksiz, pad):
        super(BasicConvModel, self).__init__()
        self.I = InputLayer()
        self.C1 = TensorflowConvLayer(inc, outc, ksiz, stride=1, padding=pad, activation_fn=nn.Tanh())
        self.C2 = TensorflowConvLayer(outc, inc, ksiz, stride=1, padding=pad, activation_fn=nn.Sigmoid())
    
    def forward(self, x):
        x = self.I(x)
        x = self.C1(x)
        x = self.C2(x)
        return x
    
class DeeperConvModel(nn.Module):
    def __init__(self):
        super(DeeperConvModel, self).__init__()
        self.I = InputLayer()
        self.C1 = TensorflowConvLayer(3, 3, 5, stride=1, padding="same", activation_fn=nn.Tanh())
        self.C2 = TensorflowConvLayer(3, 6, 5, stride=1, padding="same", activation_fn=nn.Tanh())
        self.C3 = TensorflowConvLayer(6, 3, 5, stride=1, padding="same", activation_fn=nn.Tanh())
        self.C4 = TensorflowConvLayer(3, 3, 5, stride=1, padding="same", activation_fn=nn.Sigmoid())
    
    
    def forward(self, x):
        x = self.I(x)
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        return x
    
class BasicConvClassifier(nn.Module):
    def __init__(self, inc):
        super(BasicConvClassifier, self).__init__()
        self.I = InputLayer()
        self.C1 = TensorflowConvLayer(inc, 16, kernel_size=5, stride=1, padding="same", activation_fn=nn.ReLU())
        self.M1 = TensorflowMaxPool2d(kernel_size=2, stride=2)
        self.C2 = TensorflowConvLayer(16, 32, kernel_size=5, stride=1, padding="same", activation_fn=nn.ReLU())
        self.M2 = TensorflowMaxPool2d(kernel_size=2, stride=2)
        self.F = nn.Flatten()
        self.L1 = TensorflowLinearLayer(32*8*8, 120, activation_fn=nn.ReLU())
        self.L2 = TensorflowLinearLayer(120, 10, activation_fn=nn.Sigmoid())

    def forward(self, x):
        x = self.I(x)
        x = self.C1(x)
        x = self.M1(x)
        x = self.C2(x)
        x = self.M2(x)
        x = self.F(x)
        x = self.L1(x)
        x = self.L2(x)
        return x

class BasicLinearModel(nn.Module):
    def __init__(self, inshape, outshape):
        super(BasicLinearModel, self).__init__()
        self.I = InputLayer()
        self.L1 = TensorflowLinearLayer(inshape, outshape, activation_fn=nn.Tanh())
        self.L2 = TensorflowLinearLayer(outshape, inshape, activation_fn=nn.Identity())
    def forward(self, x):
        x = self.I(x)
        x = self.L1(x)
        x = self.L2(x)
        return x
    
class DeeperLinearModel(nn.Module):
    def __init__(self, inshape, midshape, outshape):
        super(DeeperLinearModel, self).__init__()
        self.I = InputLayer()
        self.L1 = TensorflowLinearLayer(inshape, midshape, activation_fn=nn.Tanh())
        self.L2 = TensorflowLinearLayer(midshape, outshape, activation_fn=nn.Tanh())
        self.L3 = TensorflowLinearLayer(outshape, midshape, activation_fn=nn.Tanh())
        self.L4 = TensorflowLinearLayer(midshape, inshape, activation_fn=nn.Sigmoid())
    def forward(self, x):
        x = self.I(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        return x


class BasicLinearClassifier(nn.Module):
    def __init__(self, inshape, midshape, classes):
        super(BasicLinearClassifier, self).__init__()
        self.I = InputLayer()
        self.L1 = TensorflowLinearLayer(inshape, midshape, activation_fn=nn.Tanh())
        self.L2 = TensorflowLinearLayer(midshape, classes, activation_fn=nn.Sigmoid())
    def forward(self, x):
        x = self.I(x)
        x = self.L1(x)
        x = self.L2(x)
        return x