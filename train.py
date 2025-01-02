import os
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pickle

from models import *

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description="Template")

parser.add_argument("-gpu", "--GPU-index", default=0, type=int, help="gpu index")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--k", default = 64, type=int, help="CNN width")
parser.add_argument("-bs", "--batch-size", default = 128, type=int, help="batch size")
parser.add_argument("--n-epoch", default = 200, type=int, help="number of epochs")
parser.add_argument("-m", "--model", default = "CNNSmall", type=str, help="the model type to select")
parser.add_argument("-ds", "--dataset", default = "CIFAR10", type=str, help="the types of datasets")

options = parser.parse_args()
device = torch.device(f'cuda:{options.GPU_index}')

np.random.seed(options.seed)
torch.random.manual_seed(options.seed)

print("Training %s on %s with: K = %d, Epochs = %d, Seed = %d"%(
    options.model, options.dataset, options.k, options.n_epoch, options.seed))


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

if not os.path.exists("./data"):
    os.mkdir("./data")
if not os.path.exists("./models"):
    os.mkdir("./models")
if not os.path.exists("./logs"):
    os.mkdir("./logs")

if options.dataset == "CIFAR10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=options.batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=options.batch_size,
                                             shuffle=True, num_workers=2)
    n_class = 10
elif options.dataset == "CIFAR100":
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=options.batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=options.batch_size,
                                             shuffle=True, num_workers=2)
    n_class = 100
    
   
model_func = globals()[options.model]
model = model_func(k=options.k, n_class = n_class).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
lambda1 = lambda epoch: 1/np.sqrt(epoch+1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [lambda1])

    
Loss_train = []
Loss_test = []
Acc_train = []
Acc_test = []
start = time.time()
for epoch in range(options.n_epoch):  # loop over the dataset multiple times
    
    Loss = []
    Acc = []
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        Loss.append(loss.item())
        Acc.append((outputs.argmax(1) == labels).float().mean().item())
        
    Loss_train.append(sum(Loss)/len(Loss))
    Acc_train.append(sum(Acc)/len(Acc))
    
    scheduler.step()
    
    Loss = []
    Acc = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            Loss.append(loss.item())
            Acc.append((outputs.argmax(1) == labels).float().mean().item())
        
    Loss_test.append(sum(Loss)/len(Loss))
    Acc_test.append(sum(Acc)/len(Acc))
    
    print("Seed: [%d], k: [%d], Epoch: [%d/%d] LR: %.4f, Loss: [%.4f/%.4f], Acc: [%.4f/%.4f], T:[%.2f]"%(
        options.seed, options.k, epoch+1,options.n_epoch,scheduler.get_last_lr()[0],Loss_test[-1],Loss_train[-1],Acc_test[-1],Acc_train[-1],time.time()-start
    ))
    
    
Loss_test = np.array(Loss_test)
Loss_train = np.array(Loss_train)
Acc_test = np.array(Acc_test)
Acc_train = np.array(Acc_train)

logs = {
    "LossTest": Loss_test,
    "LossTrain": Loss_train,
    "AccTest": Acc_test,
    "AccTrain": Acc_train,
}

with open(f"logs/{options.dataset}_{options.model}_K{options.k}_{options.seed}.p", "wb") as f:
    pickle.dump(logs, f)
    
np.savetxt(f"logs/Loss_test_{options.dataset}_{options.model}_K{options.k}_{options.seed}.txt", Loss_test)
np.savetxt(f"logs/Loss_train_{options.dataset}_{options.model}_K{options.k}_{options.seed}.txt", Loss_train)
np.savetxt(f"logs/Acc_test_{options.dataset}_{options.model}_K{options.k}_{options.seed}.txt", Acc_test)
np.savetxt(f"logs/Acc_train_{options.dataset}_{options.model}_K{options.k}_{options.seed}.txt", Acc_train)


torch.save(model.state_dict(), "models/%ss_K%d_epoch%d_seed%d"%(options.model, options.k, (epoch+1), options.seed))