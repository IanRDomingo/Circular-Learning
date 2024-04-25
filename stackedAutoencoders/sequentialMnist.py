from utils.CircAE import *
from utils.classes import *
from torchvision import transforms
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import random
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

def main(args):
    INPUT_SIZE = 784
    BATCHSIZE = args.batchsize
    LEARNING_RATE_CLASSIFIER = args.learning_rate
    TOURBILLON_SIZES = args.hidden_sizes
    seed = args.seed
    set_seed(seed)
    device = args.device
    models = []
    models.append(BasicLinearModel(INPUT_SIZE, TOURBILLON_SIZES[0]).to(device))
    for i in range(len(TOURBILLON_SIZES)-1):
        models.append(BasicLinearModel(TOURBILLON_SIZES[i], TOURBILLON_SIZES[i+1]).to(device))

    cifar10_train = MNIST(BATCHSIZE)
    d_train = cifar10_train.dataset
    cifar10_test = MNIST(BATCHSIZE, test=True)
    d_test = cifar10_test.dataset

    for i in range(len(TOURBILLON_SIZES)):
        pre = []
        for j in range(i):
            pre.append(models[j].L1)
        cir = CircularAE(models[i], ncirc=1, no_target_flag=False, t2_minus_t1=False, device = device)
        m, hist = cir.train(dataset=d_train,
                                    epochs=100, valid_dataset=d_test, train_bias=True,
                                    train_bias_out=True, verbose=True, predecessors = pre)

    model = ClassifierNetwork(TOURBILLON_SIZES[len(TOURBILLON_SIZES)-1], 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_CLASSIFIER)

    hist = {}
    hist['test_acc'] = []
    hist['test_loss'] = []
    hist['train_loss'] = []
    hist['train_acc'] = []

    pre = []
    for j in range(len(TOURBILLON_SIZES)):
        pre.append(models[j].L1)

    for epoch in range(100):
        train_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(d_train):
            images = images.to(device)
            for predecessor in pre:
                images = predecessor(images)
            labels = labels.to(device).float()

            outputs = model(images).float()
            loss = criterion(outputs, labels)
            train_loss+=loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, label = torch.max(labels.data, 1)
            correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{100}], Loss: {train_loss / len(d_train):.4f}')
        hist['train_loss'].append(train_loss / len(d_train))
        hist['train_acc'].append(100 * correct / total)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            for images, labels in d_test:
                images = images.to(device)
                for predecessor in pre:
                    images = predecessor(images)
                labels = labels.to(device).float()
                outputs = model(images).float()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, label = torch.max(labels.data, 1)
                correct += (predicted == label).sum().item()

                loss = criterion(outputs, labels)
                test_loss += loss.item()

            avg_loss = test_loss / len(d_test)

            hist['test_loss'].append(avg_loss)
            hist['test_acc'].append(100 * correct / total)

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
            print(f'Loss of the network on the 10000 test images: {avg_loss:.4f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a stack of autoencoders using recirculation asynchronously')
    parser.add_argument('--batchsize', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the classifier (default: 0.001)')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, help='Hidden hinge layer sizes of the stacked autoencoders (default: [256])', default=[256])
    parser.add_argument('--seed', type = int, default = 101, help = 'Seed used for reproducibility (default: 101)')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Device used to run program (default: cpu)')
    args = parser.parse_args()
    sys.exit(main(args))