from torch.utils.data import Dataset
from typing import List


class ImageMatchingDataset(Dataset):

    
    def __init__(self, dataset: List[str]):
        self.dataset = dataset # A list of image path


    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample
    
    
    def __len__(self):
        length = len(self.dataset)
        return length