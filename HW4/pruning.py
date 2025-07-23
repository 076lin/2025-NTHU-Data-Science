import argparse
import os
from tqdm import tqdm
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from net.models import AlexNet
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--re-epochs', type=int, default=30, metavar='N',
                    help='number of epochs to retrain (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--load-model-path', type=str,
                    default='saves/initial_model.ptmodel', help='path to load pre-trained model')
parser.add_argument('--save-model-path', type=str,
                    default='saves/model_after_retraining.ptmodel', help='path to load pre-trained model')
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!')

# Loader
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True, transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
model = AlexNet().to(device)
print(model)
util.print_model_parameters(model)

# Turn off "SourceChangeWarning" of Pytorch
warnings.filterwarnings("ignore")


def train(epochs, optimizer):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            for name, p in model.named_parameters():

                #################################
                # TODO:
                #    zero-out all the gradients corresponding to the pruned weights
                #################################
                mask = (p.data != 0).float()
                p.grad.data.mul_(mask)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(
                    f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  '
                    f'Loss: {loss.item():.6f}'
                )


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # Get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


def main():
    global model

    # -------------------------------
    # Initial training (Train from scratch)
    # -------------------------------
    print("--- Initial training ---")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    train(args.epochs, optimizer)
    torch.save(model, f"saves/initial_model.ptmodel")
    accuracy = test()
    util.log(f"Initial_accuracy {accuracy}", args.log)
    util.print_nonzeros(model)

    # -------------------------------
    # Initial training (Use pre-trained model)
    # -------------------------------
    model = torch.load(args.load_model_path, weights_only=False)
    accuracy = test()
    util.log(f"Initial_accuracy {accuracy}", args.log)
    util.print_nonzeros(model)

    # Pruning
    # model.prune_by_percentile()
    model.prune_by_std(args.sensitivity)
    print("--- After pruning ---")
    accuracy = test()
    util.log(f"Accuracy_after_pruning {accuracy}", args.log)
    util.print_nonzeros(model)

    # Retrain
    print("--- Retraining ---")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    train(args.re_epochs, optimizer)
    torch.save(model, args.save_model_path)
    print("--- After Retraining ---")
    util.print_nonzeros(model)
    accuracy = test()
    util.log(f"Accuracy_after_retraining {accuracy}", args.log)


if __name__ == '__main__':
    main()


