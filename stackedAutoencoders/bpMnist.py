from utils.BP import *
from utils.classes import *
from torchvision import transforms
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import random
import sys
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, out_size):
        super(ClassifierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, out_size)

    def forward(self, x):
        return self.fc1(x)

class MNIST:
    def __init__(self, batchsize, test=False, norm=True):
        transform_list = [transforms.ToTensor()]
        if norm:
            transform_list.append(transforms.Normalize((0.5,), (0.5,))) 
        
        transform = transforms.Compose(transform_list)

        cifar10_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        cifar10_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        x_train = cifar10_train.data / 255.0 if norm else cifar10_train.data
        x_test = cifar10_test.data / 255.0 if norm else cifar10_test.data
        y_train = F.one_hot(cifar10_train.targets, num_classes=10)
        y_test = F.one_hot(cifar10_test.targets, num_classes=10)

        x_train = torch.tensor(x_train, dtype=torch.float32)
        #x_train = x_train.reshape(-1, 1, 28, 28)
        x_train = x_train.view(-1, 784)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        #x_test = x_test.reshape(-1, 1, 28, 28)
        x_test = x_test.view(-1, 784)

        dtrain = torch.utils.data.TensorDataset(x_train, y_train)
        dtest = torch.utils.data.TensorDataset(x_test, y_test)

        if not test:
            self.dataset = DataLoader(dtrain, batch_size=batchsize, shuffle=True)
            self.ds = x_train
            self.dt = y_train
        else:
            self.dataset = DataLoader(dtest, batch_size=batchsize, shuffle=True)
            self.ds = x_test
            self.dt = y_test

class BasicLinearClassifierMnist(nn.Module):
    def __init__(self, inshape, midshape1, midshape2, classes):
        super(BasicLinearClassifierMnist, self).__init__()
        self.I = InputLayer()
        self.L1 = TensorflowLinearLayer(inshape, midshape1, activation_fn=nn.ReLU())
        self.L2 = TensorflowLinearLayer(midshape1, midshape2, activation_fn=nn.ReLU())
        self.L4 = TensorflowLinearLayer(midshape2, classes, activation_fn=nn.Sigmoid())
    def forward(self, x):
        x = self.I(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L4(x)
        return x

def main(args):
    BATCHSIZE = args.batchsize
    LEARNING_RATE_CLASSIFIER = args.learning_rate_classifier
    seed = args.seed
    set_seed(seed)
    device = args.device

    cifar10_train = MNIST(BATCHSIZE)
    d_train = cifar10_train.dataset
    cifar10_test = MNIST(BATCHSIZE, test=True)
    d_test = cifar10_test.dataset
    
    model = BasicLinearClassifierMnist(784, 624, 520, 10).to(device)
    rbp = BP(model, False, device=device, init = 'wt')
    model, hist = rbp.train(d_train, lr = LEARNING_RATE_CLASSIFIER, epochs=100, valid_dataset=d_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a stack of autoencoders using recirculation asynchronously')
    parser.add_argument('--batchsize', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--learning_rate_classifier', type=float, default=0.001, help='Learning rate for the classifier (default: 0.001)')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Device used to run program (default: cpu)')
    parser.add_argument('--seed', type = int, default = 101, help = 'Seed used for reproducibility (default: 101)')
    args = parser.parse_args()
    sys.exit(main(args))