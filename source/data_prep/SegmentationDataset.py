from PIL import Image
import torch
from torch.utils.data import Dataset
from io import BytesIO
import numpy as np
class SegmentationDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(BytesIO(self.dataframe.iloc[idx, 0]['bytes'])) # image
        mask = self.convert2SimpleMask(np.array(Image.open(BytesIO(self.dataframe.iloc[idx, 1]['bytes'])))) # mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def convert2SimpleMask(self, mask: np.array):
        mask[np.isin(mask, [1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17])] = 1
        mask[np.isin(mask, [2, 11, 12, 13, 14, 15])] = 2
        return mask