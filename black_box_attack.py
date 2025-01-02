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

### FGSM attack

def fgsm_attack(model, image, label, epsilon = 0.3, grad = None):
    
    if grad is None:
        model.zero_grad()
        image.requires_grad = True
        output = model(image)
        prob = torch.softmax(output,dim=1)[range(options.batch_size), label].sum()
        prob.backward()
    
        grad = image.grad.data.detach().clone()
    
    sign_grad = grad.sign()
    perturbed_image = image - epsilon * sign_grad
    perturbed_image = torch.clamp(perturbed_image, image.min(), image.max())
    model.zero_grad()
    
    return perturbed_image





avg_performance = []
for source in models:
    performance = []
    for idx, data in enumerate(testloader, 0):
        if idx == n_batch:
            break
        source.zero_grad()
        image, label = data
        image = image.to(device)
        label = label.to(device)
        
        perturbed_image = fgsm_attack(
            source, image.detach().clone(), label, 
            epsilon = 0.05)
        
        preds_raw = []
        preds_attacked = []
        with torch.no_grad():
            for target in models:
                preds_raw.append(target(image).detach())
                preds_attacked.append(target(perturbed_image).detach())
 
        preds_raw = torch.stack(preds_raw)
        preds_attacked = torch.stack(preds_attacked)
        preds_raw = torch.softmax(preds_raw, dim=2)
        preds_attacked = torch.softmax(preds_attacked,dim=2)
        preds_raw = preds_raw[:, range(options.batch_size), label]
        preds_attacked = preds_attacked[:, range(options.batch_size), label]
        
        performance.append(((preds_attacked/preds_raw).mean(1).cpu().detach().numpy()))
        
    avg_performance.append(sum(performance)/n_batch)
    
avg_performance = np.array(avg_performance)
avg_performance[range(len(ks)), range(len(ks))] = np.nan


plt.figure(figsize = (5,5), dpi=200)
plt.imshow(avg_performance, cmap="jet", vmin=0,vmax=1)
plt.yticks(range(len(ks)), ks, fontsize = 6)
plt.xticks(range(len(ks)), ks, rotation=60, fontsize = 6)
plt.xlabel(r"Model Size $k_1$")
plt.ylabel(r"Model Size $k_2$")
plt.colorbar(shrink=0.8)
plt.savefig("black-box_attack.png", bbox_inches="tight")
plt.show()

print("The black-box attack map is saved in black-box_attack.png!")