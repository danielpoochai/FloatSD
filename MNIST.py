from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import floatSD_utility as SD
import floatSD_cuda
import quant_function as QF
from torchvision import datasets, transforms 

momemtum = 0.9
exp_off = -3
Loss_list = []
Accuracy_list = []
iter_list = []
iter = 0

# Quantizer
quant = lambda : QF.Quantizer()

def quant_fp8(x):
    # x_shape = x.shape
    # x = x.reshape(-1)
    x = QF.float_quantize(x)
    # x = x.reshape(x_shape)
    return x
    
class Net(nn.Module):
    def __init__(self, quant):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        self.quant = quant()

        # velocity (with momentum)
        self.conv1_vel = torch.cuda.FloatTensor(self.conv1.weight.shape)
        self.conv2_vel = torch.cuda.FloatTensor(self.conv2.weight.shape)
        self.fc1_vel = torch.cuda.FloatTensor(self.fc1.weight.shape)
        self.fc2_vel = torch.cuda.FloatTensor(self.fc2.weight.shape)
        # define floatSD master copy and initialize
        self.conv1_copy = SD.floatSD(self.conv1.weight, exp_off)
        self.conv2_copy = SD.floatSD(self.conv2.weight, exp_off)
        self.fc1_copy = SD.floatSD(self.fc1.weight, exp_off)
        self.fc2_copy = SD.floatSD(self.fc2.weight, exp_off)
        # update FP32 values(in cuda)
        self.conv1.weight = torch.nn.Parameter(self.conv1_copy.fp32_weight)
        self.conv2.weight = torch.nn.Parameter(self.conv2_copy.fp32_weight)
        self.fc1.weight = torch.nn.Parameter(self.fc1_copy.fp32_weight)
        self.fc2.weight = torch.nn.Parameter(self.fc2_copy.fp32_weight)
        # print(self.conv1_copy.master_copy)
        # print(self.conv1_copy.fp32_weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.quant(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.quant(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.quant(x)
        x = self.fc2(x)
        return x
    
def train(args, model, device, train_loader, test_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        

        loss.backward()

        model.conv1.weight.grad = QF.float_quantize(model.conv1.weight.grad)
        model.conv2.weight.grad = QF.float_quantize(model.conv2.weight.grad)
        model.fc1.weight.grad = QF.float_quantize(model.fc1.weight.grad)
        model.fc2.weight.grad = QF.float_quantize(model.fc2.weight.grad)
        for p in model.parameters():
            p.grad = model.quant(p.grad)
        
        optimizer.step()

        # STU
        global iter
        lr = 0.01*pow((1+0.0001*iter), -0.75)
        
        # conv1
        conv1_new_copy = floatSD_cuda.STU( lr, momemtum, exp_off, model.conv1_vel,
            model.conv1.weight.grad, model.conv1_copy.master_copy
        )
        conv1_new_copy = np.array(conv1_new_copy)
        model.conv1_copy.master_copy = conv1_new_copy[0]
        model.conv1_vel = conv1_new_copy[1]
        # conv2
        conv2_new_copy = floatSD_cuda.STU( lr, momemtum, exp_off, model.conv2_vel,
            model.conv2.weight.grad, model.conv2_copy.master_copy
        )
        conv2_new_copy = np.array(conv2_new_copy)
        model.conv2_copy.master_copy = conv2_new_copy[0]
        model.conv2_vel = conv2_new_copy[1]
        # fc1
        fc1_new_copy = floatSD_cuda.STU( lr, momemtum, exp_off, model.fc1_vel,
            model.fc1.weight.grad, model.fc1_copy.master_copy
        )
        fc1_new_copy = np.array(fc1_new_copy)
        model.fc1_copy.master_copy = fc1_new_copy[0]
        model.fc1_vel = fc1_new_copy[1]
        # fc2
        fc2_new_copy = floatSD_cuda.STU( lr, momemtum, exp_off, model.fc2_vel,
            model.fc2.weight.grad, model.fc2_copy.master_copy
        )
        fc2_new_copy = np.array(fc2_new_copy)
        model.fc2_copy.master_copy = fc2_new_copy[0]
        model.fc2_vel = fc2_new_copy[1]
        
        # update fp32 weight 
        model.conv1_copy.fp32_weight = floatSD_cuda.floatSD_quant(exp_off, model.conv1_copy.master_copy, model.conv1_copy.fp32_weight)   
        model.conv2_copy.fp32_weight = floatSD_cuda.floatSD_quant(exp_off, model.conv2_copy.master_copy, model.conv2_copy.fp32_weight)
        model.fc1_copy.fp32_weight = floatSD_cuda.floatSD_quant(exp_off, model.fc1_copy.master_copy, model.fc1_copy.fp32_weight)
        model.fc2_copy.fp32_weight = floatSD_cuda.floatSD_quant(exp_off, model.fc2_copy.master_copy, model.fc2_copy.fp32_weight)
        model.conv1.weight = torch.nn.Parameter(model.conv1_copy.fp32_weight)
        model.conv2.weight = torch.nn.Parameter(model.conv2_copy.fp32_weight)
        model.fc1.weight = torch.nn.Parameter(model.fc1_copy.fp32_weight)
        model.fc2.weight = torch.nn.Parameter(model.fc2_copy.fp32_weight)
        if iter < 3:
            print(model.conv1_copy.master_copy)
            print(model.conv1_copy.fp32_weight)
        
        # testing per 50 iter
        iter = iter +1
        if iter % 100 == 0:
            test(args, model, device, test_loader)
            
        if iter > 10000:
            break
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
            # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    # record acc loss
    global iter
    Loss_list.append(test_loss)
    Accuracy_list.append(100 * correct / (len(test_loader.dataset)))
    iter_list.append(iter)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    # loading data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # initialize model
    model = Net(quant).to(device)
    
    # define optimizer
    global iter
    lr = 0.01*pow((1+0.0001*iter), -0.75)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, 20):
        train(args, model, device, train_loader, test_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        if iter > 10000:
            break
    
    # plot acc loss
    plt.plot(iter_list, Accuracy_list)
    plt.title('Test accuracy')
    plt.ylabel('Test accuracy')
    plt.xlabel('batch')
    plt.savefig("accuracy_mnist.jpg")
    plt.show()
    
    plt.plot(iter_list, Loss_list)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('batch')
    plt.savefig("loss_mnist.jpg")
    plt.show()
    
    #save result
    np.save('iter', iter_list)
    np.save('acc_MNIST_orig', Accuracy_list)
    np.save('loss_MNIST_orig', Loss_list)
    
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
