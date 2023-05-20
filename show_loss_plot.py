import numpy as np
import matplotlib.pyplot as plt
import glob 

for i in range(len(glob.glob("./Results/model_loss*"))):
    values = np.load(f'./Results/model_loss_{i}.npy')
    keys = [str(i) for i in range(len(values))]

    plt.plot(values)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"LOSS - Model in epoch {i}")
    plt.savefig(f'./Results_images/LOSS_PLOT/LOSS - Model in epoch {i}.png')
    plt.clf()
