import numpy as np
import matplotlib.pyplot as plt
import glob 

for i in range(len(glob.glob("./Results/model_acc*"))):
    values = np.load(f'./Results/model_acc_{i}.npy')
    keys = [str(i) for i in range(len(values))]

    plt.bar(keys, values, color ='blue', width = 0.75)
    plt.axhline(y=0, color='r', linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"ACC - Model in epoch {i}")
    plt.savefig(f'./Results_images/ACC/ACC - Model in epoch {i}.png')
    plt.clf()