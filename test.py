import glob
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

from adversarial import PGD
from dataset import getTestset
from models import loadModel

MODELS_PATH = "./Models/*.pt"
DEVICE = "cuda:0"
BATCH_SIZE_TEST = 512
NOM_WORKERS_TEST = 4

models = glob.glob(MODELS_PATH)

def myFunc(e):
    if(e.split("/")[-1].split("_")[-1].split(".")[0] == "fin"):
        return 1000000
        
    return int(e.split("/")[-1].split("_")[-1].split(".")[0])

models.sort(reverse=False, key=myFunc)

criterion = nn.CrossEntropyLoss()

image_names = []
label_names = []

test_set = getTestset(BATCH_SIZE_TEST, NOM_WORKERS_TEST, PIN_MEMORY=True)
image_, label_ = next(iter(test_set))

for i in tqdm(range(len(models))):
    m = models[i]
    image = image_.clone()
    label = label_.clone()

    model = loadModel(m).to(DEVICE)

    image_adversarial = PGD(
        image.to(DEVICE),
        label.to(DEVICE),
        model,
        epsilon=8/255,
        stepSize=2/255,
        lossFun=nn.CrossEntropyLoss(),
        iterationNumber=10
    )

    torch.save(image_adversarial, f"./DataStore/image_{i}.pt")
    torch.save(label, f"./DataStore/label_{i}.pt")
    
    image_names.append(f"./DataStore/image_{i}.pt")
    label_names.append(f"./DataStore/label_{i}.pt")

criterion = nn.CrossEntropyLoss()

for i in tqdm(range(len(models))):
    m = models[i]
    
    ACC = []
    LOSS = []

    model = loadModel(m).to(DEVICE)

    for k in range(len(image_names)):
        image_o = image_.clone().to(DEVICE)
        image_o = image_o.to(DEVICE)

        image = torch.load(image_names[k]).to(DEVICE)
        label = torch.load(label_names[k]).to(DEVICE)

        output = model(image)

        loss = criterion(output, label).item()
        _, predicted = torch.max(output.data, 1)

        total = label.size(0)
        correct = (predicted == label).sum().item()

        output_o = model(image_o)

        loss_o = criterion(output_o, label).item()
        _, predicted_o = torch.max(output_o.data, 1)

        total = label.size(0)
        correct_o = (predicted_o == label).sum().item()

        ACC.append((correct_o / total) - (correct / total))
        LOSS.append(loss - loss_o)
        
    with open(f'./Results/model_acc_{i}.npy', 'wb') as f:
        np.save(f, ACC)

    with open(f'./Results/model_loss_{i}.npy', 'wb') as f:
        np.save(f, LOSS)
