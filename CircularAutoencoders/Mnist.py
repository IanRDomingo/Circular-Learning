import sys
from utils.classes import *
from utils.BP import *
from utils.RBP import *
from utils.CircAE import *
import random
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

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

        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_train = x_train.view(-1, 784)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        x_test = x_test.view(-1, 784)

        if not test:
            self.dataset = DataLoader(x_train, batch_size=batchsize, shuffle=True)
        else:
            self.dataset = DataLoader(x_test, batch_size=batchsize, shuffle=True)


def build_lin_model():
    m = BasicLinearModel(784, 256)
    return m

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    device = args.device
    model = build_lin_model().to(device)
    set_seed(args.seed)

    batchsize = args.batchsize

    cifar10_train = MNIST(batchsize)
    d_train = cifar10_train.dataset

    cifar10_test = MNIST(batchsize, test=True)
    d_test = cifar10_test.dataset

    if args.training_type == 'BP':
        cir = BP(model, autoencoding=True, device = device)
        model, hist = cir.train(dataset=d_train, lr=args.learning_rate,
                                epochs=100, valid_dataset=d_test)
    elif args.training_type == 'RBP':
        cir = RBP(model, autoencoding=True, device = device)
        model, hist = cir.train(dataset=d_train, lr=args.learning_rate,
                                epochs=100, valid_dataset=d_test)
    elif args.training_type == 'Recirculation':
        cir = CircularAE(model, ncirc=args.ncirc, device = device)
        model, hist = cir.train(dataset=d_train, lr=args.learning_rate,
                                epochs=100, valid_dataset=d_test)
    else:
        raise ValueError("invalid training type")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a stack of autoencoders using recirculation asynchronously')
    parser.add_argument('--batchsize', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the classifier (default: 0.001)')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Device used to run program (default: cpu)')
    parser.add_argument('--seed', type = int, default = 101, help = 'Seed used for reproducibility (default: 101)')
    parser.add_argument('--ncirc', type = int, default = 1, help = 'Number of CAE recirculations (default: 1)')
    parser.add_argument('training_type', type = str, help = 'Training Type for the model (Recirculation, RBP, BP)')
    args = parser.parse_args()
    sys.exit(main(args))