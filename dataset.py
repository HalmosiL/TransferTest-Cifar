import torchvision.transforms as transforms
import torchvision
import torch

transform_test =  transforms.Compose([
    transforms.ToTensor(),
])

def getTestset(BATCH_SIZE_TEST, NOM_WORKERS_TEST, PIN_MEMORY=False):
    testset = torchvision.datasets.CIFAR10(
        root='./Data',
        train=False,
        download=True,
        transform=transform_test
    )

    return torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=NOM_WORKERS_TEST,
        pin_memory=PIN_MEMORY
    )