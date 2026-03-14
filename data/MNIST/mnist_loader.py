import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MNIST(Dataset):
    """
    A standalone MNIST loader that reads raw .gz files to bypass 
    torchvision download and folder-structure versioning issues.
    """
    def __init__(self, root='./data', train=True, transform=None):
        self.root = root
        
        # Default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = transform
        
        prefix = "train" if train else "t10k"
        raw_path = os.path.join(self.root, "MNIST", "raw")
        img_path = os.path.join(raw_path, f"{prefix}-images-idx3-ubyte.gz")
        lbl_path = os.path.join(raw_path, f"{prefix}-labels-idx1-ubyte.gz")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"MNIST files not found at {os.path.abspath(raw_path)}")

        with gzip.open(img_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            self.data = torch.from_numpy(data.copy()).view(-1, 28, 28)
            
        with gzip.open(lbl_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
            self.targets = torch.from_numpy(labels.copy()).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target