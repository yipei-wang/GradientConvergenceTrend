import os
import numpy as np
import torch
import torchvision
from models import CNNSmall
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser(description="Template")

parser.add_argument("-gpu", "--GPU-index", default=0, type=int, help="gpu index")
parser.add_argument("-bs", "--batch-size", default = 128, type=int, help="batch size")
parser.add_argument("--n-epoch", default = 200, type=int, help="number of epochs")
parser.add_argument("-m", "--model", default = "CNNSmall", type=str, help="the model type to select")


options = parser.parse_args()
device = torch.device(f'cuda:{options.GPU_index}')

use_p = False
down_size = 8
n_batch = 2

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

if not os.path.exists("./data"):
    os.mkdir("./data")
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=options.batch_size,
                                         shuffle=False, num_workers=2)

ks = [8, 10, 12, 14,
     16, 20, 24, 28,
     32, 40, 48, 56,
     64, 80, 96, 112,
     128]


models = []
for k in ks:
    model = CNNSmall(k=k).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(
                "models_example/CNNSmall_K%d"%k), map_location=device))        
    model.eval()
    models.append(model)
print("Finish Loading!")

### GRADIENT COSINE SIMILARITY

def cosine_similarity(grad):
    grad_flat = grad.view(len(ks),options.batch_size,-1)
    cs = torch.zeros(len(ks),len(ks))
    for i in range(len(ks)):
        for j in range(i+1,len(ks)):
            cs[i,j] = F.cosine_similarity(grad_flat[i], grad_flat[j]).sum()
    return cs


cs = torch.zeros(len(ks),len(ks)).numpy()
criterion = nn.CrossEntropyLoss()
for idx, data in enumerate(testloader, 0):
    if idx == n_batch:
        break
    grad = []
    for model in models:
       
        model.zero_grad()
        image, label = data
        image = image.to(device).requires_grad_(True)
        label = label.to(device)
        
        pred = model(image)
        
        if use_p:
            prob = torch.softmax(pred, dim=-1)
            prob[range(options.batch_size),label[range(options.batch_size)]].sum().backward()
        else:
            pred[range(options.batch_size),label[range(options.batch_size)]].sum().backward()
            
    
        image_grad = image.grad.detach().clone()
        if down_size < 32:
            image_grad = F.interpolate(image.grad.detach().clone(),(down_size,down_size), mode='area')
            
            
        grad.append(image_grad)
 
        
    grad = torch.stack(grad)
        
    with torch.no_grad():
        cs += cosine_similarity(grad).detach().cpu().numpy()
    
cs /= (n_batch*options.batch_size)

plt.figure(figsize = (5,5), dpi=200)
plt.imshow(cs, cmap="jet")
plt.yticks(range(len(ks)), ks, fontsize = 6)
plt.xticks(range(len(ks)), ks, rotation=60, fontsize = 6)
plt.xlabel(r"Model Size $k_1$")
plt.ylabel(r"Model Size $k_2$")
plt.colorbar(shrink=0.8)
plt.savefig("saliency_similarity.png", bbox_inches="tight")
plt.show()

print("The upper-triangular similarity map is saved in saliency_similarity.png!")