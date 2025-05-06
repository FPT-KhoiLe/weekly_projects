import os
from pathlib import Path
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

DEFAULT_ROOT = Path(__file__).parent.parent / "datas"

def get_loaders(batch_size: int = 64, root: str = DEFAULT_ROOT, transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)):
    train_set = MNIST(root=root, train=True, download=True, transform=transform)
    test_set = MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("MNIST dataset saved in: ", root)
    return train_loader, test_loader
