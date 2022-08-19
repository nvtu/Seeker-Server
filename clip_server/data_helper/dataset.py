from torch.utils.data import Dataset
from PIL import Image
from typing import List
import numpy as np


class ImageMatchingDataset(Dataset):

    
    def __init__(self, dataset: List[str], transform, device='cuda'):
        self.dataset = dataset # A list of image path
        self.transform = transform # Preprocessing from CLIP model
        self.device = device # Device to use (GPU or CPU)


    def __getitem__(self, index):
        path = self.dataset[index]
        image = self.transform(Image.open(path)).to(self.device)
        return image, path
    
    
    def __len__(self):
        length = len(self.dataset)
        return length