import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet_5 as LeNet
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 100)')
                    
parser.add_argument('--reg', type=int, default=0, metavar='R',
                    help='regularization type: 0:None 1:L1 2:Hoyer 3:HS 4:Transformed L1')
parser.add_argument('--decay', type=float, default=0.001, metavar='D',
                    help='weight decay for regularizer (default: 0.001)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=12345678, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--p', type=int, default=1, metavar='R',
                    help='regularization type for new approach:  1:L1 2:L2')
parser.add_argument('--q', type=int, default=1, metavar='R',
                    help='regularization type for new approach:  1:L1 2:L2')

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
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
model = LeNet(mask=False).to(device)

print(model)
util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr)
initial_optimizer_state_dict = optimizer.state_dict()


def train(epochs, decay=0, threshold=0.0):
    model.train()
    pbar = tqdm(range(epochs), total=epochs)
    curves = np.zeros((epochs,14))
    p = 1
    q = 1
    p = args.p
    q = args.q
    
    for epoch in pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            
            reg = 0.0
            if decay:
                reg = 0.0
                if args.reg==5:
                    weights_out = 0
                    start = False
                    for name, param in model.named_parameters():
                        if param.requires_grad and 'weight' in name and torch.sum(torch.abs(param))>0:
                            
                            weights_in = weights_out
                            weights_out = param  
                            
                            if start:
                                if len(weights_in.size()) == len(weights_out.size()):
                                    if len(weights_in.size()) == 2:
                                        input_weights = torch.sum(torch.abs(weights_in), dim=1)**2
                                        output_weights = torch.sum(torch.abs(weights_out), dim=0)**2
                                    elif len(weights_in.size()) == 4:
                                        input_weights = torch.sum(torch.abs(weights_in), dim=(1,2,3))**2
                                        output_weights = torch.sum(torch.abs(weights_out), dim=(0,2,3))**2
                                    reg += torch.sum(input_weights + output_weights)
                                else:
                                    reg += torch.sum(torch.sum(torch.abs(weights_out), dim=0)**2)
                            start = True
                            
                else:
                    for param in model.parameters():
                        if param.requires_grad and torch.sum(torch.abs(param))>0:
                            if args.reg==1:    
                                reg += torch.sum(torch.abs(param))
                            elif args.reg==2:
                                reg += torch.sum(torch.abs(param))/torch.sqrt(torch.sum(param**2))
                            elif args.reg==3:
                                reg += (torch.sum(torch.abs(param))**2)/torch.sum(param**2)
                            elif args.reg==4:    
                                reg += torch.sum(2*torch.abs(param)/(1+torch.abs(param)))
                            else:
                                reg = 0.0         
            total_loss = loss+decay*reg
                
            total_loss.backward()

            optimizer.step()
            

                
            
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.3f}  Reg: {reg:.3f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

if __name__ == '__main__':
    if args.pretrained:
        model.load_state_dict(torch.load('saves/elt_0.0_0.pth'))
        accuracy = test()
    # Initial training
    print("--- Initial training ---")
    train(args.epochs, decay=args.decay, threshold=0.0)
    accuracy = test()
    torch.save(model.state_dict(), 'saves/elt_'+str(args.decay)+'_'+str(args.reg)+'.pth')

    util.log(args.log, f"initial_accuracy {accuracy}")
    #util.print_nonzeros(model)


