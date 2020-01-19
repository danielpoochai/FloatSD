from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms 

from qtorch.quant import Quantizer, quantizer
from qtorch import FloatingPoint
import floatSD_utility as SD
import floatSD_cuda
import quant_function as QF

loss_iter = []
loss_avg = []
loss_test = []
acc_test = []
iteration = 0 

conv1_1_off = -2
conv1_2_off = -5        
conv2_1_off = -5
conv2_2_off = -5
conv3_1_off = -5
conv3_2_off = -6
fc_off = -6
out_off = -5

def quant_fp8(x):
    x = QF.float_quantize(x)
    return x

class VGG(nn.Module):
    def __init__(self, quant):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        # self.quant = quant()

        self.bn1_1 = nn.BatchNorm2d(128)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.bn3_2 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512*4*4, 1024)
        self.out = nn.Linear(1024, 10)

        self.drop = nn.Dropout(p=0.5)

        #quant
        self.quant = quant()

        #floatSD
        # print(self.fc.weight)
        #master copy
        self.conv1_1_copy = SD.floatSD(self.conv1_1.weight, conv1_1_off)
        self.conv1_2_copy = SD.floatSD(self.conv1_2.weight, conv1_2_off)
        self.conv2_1_copy = SD.floatSD(self.conv2_1.weight, conv2_1_off)
        self.conv2_2_copy = SD.floatSD(self.conv2_2.weight, conv2_2_off)
        self.conv3_1_copy = SD.floatSD(self.conv3_1.weight, conv3_1_off)
        self.conv3_2_copy = SD.floatSD(self.conv3_2.weight, conv3_2_off)
        self.fc_copy = SD.floatSD(self.fc.weight, fc_off)
        self.out_copy = SD.floatSD(self.out.weight, out_off)
        #velocity
        self.conv1_1_vel = torch.cuda.FloatTensor(self.conv1_1.weight.shape)
        self.conv1_2_vel = torch.cuda.FloatTensor(self.conv1_2.weight.shape)
        self.conv2_1_vel = torch.cuda.FloatTensor(self.conv2_1.weight.shape)
        self.conv2_2_vel = torch.cuda.FloatTensor(self.conv2_2.weight.shape)
        self.conv3_1_vel = torch.cuda.FloatTensor(self.conv3_1.weight.shape)
        self.conv3_2_vel = torch.cuda.FloatTensor(self.conv3_2.weight.shape)
        self.fc_vel = torch.cuda.FloatTensor(self.fc.weight.shape)
        self.out_vel = torch.cuda.FloatTensor(self.out.weight.shape)

        update FP32 values
        self.conv1_1.weight = torch.nn.Parameter(self.conv1_1_copy.fp32_weight)
        self.conv1_2.weight = torch.nn.Parameter(self.conv1_2_copy.fp32_weight)
        self.conv2_1.weight = torch.nn.Parameter(self.conv2_1_copy.fp32_weight)
        self.conv2_2.weight = torch.nn.Parameter(self.conv2_2_copy.fp32_weight)
        self.conv3_1.weight = torch.nn.Parameter(self.conv3_1_copy.fp32_weight)
        self.conv3_2.weight = torch.nn.Parameter(self.conv3_2_copy.fp32_weight)
        # self.fc.weight = torch.nn.Parameter(self.fc_copy.fp32_weight)
        self.out.weight = torch.nn.Parameter(self.out_copy.fp32_weight)
        # print(self.fc.weight)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.quant(x)

        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.quant(x)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.quant(x)

        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.quant(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.quant(x)

        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.quant(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc(x))
        x = self.quant(x)

        x = self.drop(x)
        out = self.out(x)
        x = self.quant(x)
        
        return F.log_softmax(out, dim = 1)

def train(args, model, device, train_loader, test_loader, epoch, optimizer, iteration, lr, momentum):
    model.train()

    #criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        output = model(data)

        optimizer.zero_grad()

        loss = F.nll_loss(output, target)

        loss.backward()
        #quant gradients
        model.conv1_1.weight.grad = QF.float_quantize(model.conv1_1.weight.grad)
        model.conv1_2.weight.grad = QF.float_quantize(model.conv1_2.weight.grad)
        model.conv2_1.weight.grad = QF.float_quantize(model.conv2_1.weight.grad)
        model.conv2_2.weight.grad = QF.float_quantize(model.conv2_2.weight.grad)
        model.conv3_1.weight.grad = QF.float_quantize(model.conv3_1.weight.grad)
        model.conv3_2.weight.grad = QF.float_quantize(model.conv3_2.weight.grad)
        model.fc.weight.grad = QF.float_quantize(model.fc.weight.grad)
        model.out.weight.grad = QF.float_quantize(model.out.weight.grad)

        optimizer.step()
        # print(model.out.weight)
        # print(model.out.weight.grad)
        #STU
        conv1_1_copy_new = floatSD_cuda.STU( lr, momentum, conv1_1_off, 
            model.conv1_1_vel, model.conv1_1.weight.grad, model.conv1_1_copy.master_copy
        )
        model.conv1_1_copy.master_copy = conv1_1_copy_new[0]
        model.conv1_1_vel = conv1_1_copy_new[1]

        conv1_2_copy_new = floatSD_cuda.STU( lr, momentum, conv1_2_off, 
            model.conv1_2_vel, model.conv1_2.weight.grad, model.conv1_2_copy.master_copy
        )
        model.conv1_2_copy.master_copy = conv1_2_copy_new[0]
        model.conv1_2_vel = conv1_2_copy_new[1]

        conv2_1_copy_new = floatSD_cuda.STU( lr, momentum, conv2_1_off, 
            model.conv2_1_vel, model.conv2_1.weight.grad, model.conv2_1_copy.master_copy
        )
        model.conv2_1_copy.master_copy = conv2_1_copy_new[0]
        model.conv2_1_vel = conv2_1_copy_new[1]

        conv2_2_copy_new = floatSD_cuda.STU( lr, momentum, conv2_2_off, 
            model.conv2_2_vel, model.conv2_2.weight.grad, model.conv2_2_copy.master_copy
        )
        model.conv2_2_copy.master_copy = conv2_2_copy_new[0]
        model.conv2_2_vel = conv2_2_copy_new[1]

        conv3_1_copy_new = floatSD_cuda.STU( lr, momentum, conv3_1_off, 
            model.conv3_1_vel, model.conv3_1.weight.grad, model.conv3_1_copy.master_copy
        )
        model.conv3_1_copy.master_copy = conv3_1_copy_new[0]
        model.conv3_1_vel = conv3_1_copy_new[1]

        conv3_2_copy_new = floatSD_cuda.STU( lr, momentum, conv3_2_off, 
            model.conv3_2_vel, model.conv3_2.weight.grad, model.conv3_2_copy.master_copy
        )
        model.conv3_2_copy.master_copy = conv3_2_copy_new[0]
        model.conv3_2_vel = conv3_2_copy_new[1]

        fc_copy_new = floatSD_cuda.STU( lr, momentum, fc_off, 
            model.fc_vel, model.fc.weight.grad, model.fc_copy.master_copy
        )
        model.fc_copy.master_copy = fc_copy_new[0]
        model.fc_vel = fc_copy_new[1]

        out_copy_new = floatSD_cuda.STU( lr, momentum, out_off, 
            model.out_vel, model.out.weight.grad, model.out_copy.master_copy
        )
        model.out_copy.master_copy = out_copy_new[0]
        model.out_vel = out_copy_new[1]


        #update fp32 weight
        # print(type(conv1_1_off), type(model.conv1_1_copy.master_copy), type(model.conv1_1_copy.fp32_weight))
        model.conv1_1_copy.fp32_weight = floatSD_cuda.floatSD_quant(conv1_1_off, model.conv1_1_copy.master_copy, model.conv1_1_copy.fp32_weight)
        model.conv1_2_copy.fp32_weight = floatSD_cuda.floatSD_quant(conv1_2_off, model.conv1_2_copy.master_copy, model.conv1_2_copy.fp32_weight)
        model.conv2_1_copy.fp32_weight = floatSD_cuda.floatSD_quant(conv2_1_off, model.conv2_1_copy.master_copy, model.conv2_1_copy.fp32_weight)
        model.conv2_2_copy.fp32_weight = floatSD_cuda.floatSD_quant(conv2_2_off, model.conv2_2_copy.master_copy, model.conv2_2_copy.fp32_weight)
        model.conv3_1_copy.fp32_weight = floatSD_cuda.floatSD_quant(conv3_1_off, model.conv3_1_copy.master_copy, model.conv3_1_copy.fp32_weight)
        model.conv3_2_copy.fp32_weight = floatSD_cuda.floatSD_quant(conv3_2_off, model.conv3_2_copy.master_copy, model.conv3_2_copy.fp32_weight)
        model.fc_copy.fp32_weight = floatSD_cuda.floatSD_quant(fc_off, model.fc_copy.master_copy, model.fc_copy.fp32_weight)
        model.out_copy.fp32_weight = floatSD_cuda.floatSD_quant(out_off, model.out_copy.master_copy, model.out_copy.fp32_weight)

        model.conv1_1.weight = torch.nn.Parameter(model.conv1_1_copy.fp32_weight)
        model.conv1_2.weight = torch.nn.Parameter(model.conv1_2_copy.fp32_weight)
        model.conv2_1.weight = torch.nn.Parameter(model.conv2_1_copy.fp32_weight)
        model.conv2_2.weight = torch.nn.Parameter(model.conv2_2_copy.fp32_weight)
        model.conv3_1.weight = torch.nn.Parameter(model.conv3_1_copy.fp32_weight)
        model.conv3_2.weight = torch.nn.Parameter(model.conv3_2_copy.fp32_weight)
        # model.fc.weight = torch.nn.Parameter(model.fc_copy.fp32_weight)
        model.out.weight = torch.nn.Parameter(model.out_copy.fp32_weight)

        
        iteration += 1

        loss_iter.append(float(loss))

        # if iteration % 100 == 0:
        #     test(args, model, device, test_loader)

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tIter:{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),iteration))
            loss_avg.append(sum(loss_iter[-args.log_interval:])/args.log_interval)



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
            # pred = output.argmax(dim=1, keepdim = True)
            # correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    loss_test.append(test_loss)
    acc_test.append(100 * correct/ len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(1)

    device = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    # loading data
    DOWNLOAD_CIFAR10 = True if not (os.path.exists('./CIFAR10/')) or not os.listdir('./CIFAR10/') else False
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023,0.1994,0.2010)),
    ])

    trainset = datasets.CIFAR10(root = './CIFAR10/', train=True, download = DOWNLOAD_CIFAR10, transform = transform_train)
    testset = datasets.CIFAR10(root = './CIFAR10/', train=False, download = DOWNLOAD_CIFAR10, transform = transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers = 4)

    quant = lambda: QF.Quantizer()

    model = VGG(quant).to(device)
    print(model)


    for epoch in range(1, args.epochs):
        if epoch < 60: 
            LR = 0.001
            MOMENT = 0.85
            optimizer = optim.SGD(model.parameters(), 0.001, momentum = 0.85 ,weight_decay=0.004)
        elif (epoch >= 60 and epoch < 70): 
            LR = 0.0001
            MOMENT = 0.89
            optimizer = optim.SGD(model.parameters(), 0.0001, momentum = 0.89 ,weight_decay=0.004)
            print("epoch!! over 60")
        elif (epoch >=70 and epoch < 80): 
            LR = 0.00005
            MOMENT = 0.9
            optimizer = optim.SGD(model.parameters(), 0.00005, momentum = 0.9 ,weight_decay=0.004)
            print("epoch!! over 70")
        else:
            LR = 0.00003
            MOMENT = 0.85
            optimizer = optim.SGD(model.parameters(), 0.00003, momentum = 0.85 ,weight_decay=0.004)

        #optimizer = optim.SGD(model.parameters(), LR, momentum = 0.9 ,weight_decay=0.004)
        #print(optimizer)

        train(args, model, device, train_loader, test_loader, epoch, optimizer, iteration, LR, MOMENT)
        test(args, model, device, test_loader)
        
    acc_save = np.array(acc_test).astype("float32")
    loss_iter_save = np.array(loss_iter).astype("float32")
    loss_avg_save = np.array(loss_avg).astype("float32")
    loss_test_save = np.array(loss_test).astype("float32")

    np.save('accuracy_cifar_quant_only.npy', acc_save)
    np.save('loss_iter_cifar_quant_only.npy', loss_iter_save)
    np.save('loss_avg_cifar_quant_only.npy', loss_avg_save)
    np.save('loss_test_cifar_quant_only.npy', loss_test_save)

    if(args.save_model):
        torch.save(model.state_dict(), "cifar.pt")

if __name__ == '__main__':
    main()
