import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def print_nonzeros(model):
    nonzero = total = 0
    flops = 0
    past = ''
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        if 'weight' in name:
            tensor = np.abs(tensor)
            if 'conv' in name:
                past = 'conv'
                dim0 = np.sum(np.sum(tensor, axis=0),axis=(1,2))
                dim1 = np.sum(np.sum(tensor, axis=1),axis=(1,2))
                kernel_height = tensor.shape[2]
                kernel_width = tensor.shape[3]
                nz_count0 = np.count_nonzero(dim0)
                nz_count1 = np.count_nonzero(dim1)
                insize = nz_count1
            if 'fc' in name:
                past = 'conv'
                dim0 = np.sum(tensor, axis=0)
                dim1 = np.sum(tensor, axis=1)
                nz_count0 = np.count_nonzero(dim0)
                nz_count1 = np.count_nonzero(dim1)                
            
            print(f'{name:20} | dim0 = {nz_count0:7} / {len(dim0):7} ({100 * nz_count0 / len(dim0):6.2f}%) | dim1 = {nz_count1:7} / {len(dim1):7} ({100 * nz_count1 / len(dim1):6.2f}%)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def print_activation(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        if 'act' in name:
            dim0 = np.prod(tensor.shape)
            nz_count0 = np.count_nonzero(tensor)
            print(f'{name:20} | dim0 = {nz_count0:7} / {dim0:7} ({100 * nz_count0 / dim0:6.2f}%)')
    #print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')



def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
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

def cal_model_flops(model, input):
    model.eval()
    input = torch.ones(input, dtype=torch.float32).cuda()
    flops_list = []

    def conv_hook(self, input, output):
        tensor = self.weight.data.cpu().numpy()
        output_channels, output_height, output_width = output[0].size()
        new_in_channels = np.count_nonzero(np.sum(np.sum(tensor, axis=0),axis=(1,2)))
        new_out_channels = np.count_nonzero(np.sum(np.sum(tensor, axis=0),axis=(1,2)))
        
        flops = (new_out_channels/self.groups) * (self.kernel_size[0] * self.kernel_size[1] * new_in_channels/self.groups) * output_height * output_width*self.groups
        flops_list.append(flops)

    def linear_hook(self, input, output):
        tensor = self.weight.data.cpu().numpy()
        new_in_features = np.count_nonzero(np.sum(tensor, axis=0))
        new_out_features = np.count_nonzero(np.sum(tensor, axis=1))
        flops = new_in_features * new_out_features
        flops_list.append(flops)

    def adjust_input_output_channels(net):
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
        for child in children:
            adjust_input_output_channels(child)

    adjust_input_output_channels(model)
    output = model(input)
    flops = sum(flops_list)

    return flops