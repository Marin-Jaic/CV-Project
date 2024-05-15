import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from .SegmentationDataset import SegmentationDataset
import torch
from torch.utils.data import DataLoader


class Preprocess:
    def get_data_loaders(self) -> tuple:
        dataset = pq.ParquetDataset('../data')

        table = dataset.read()
        df = table.to_pandas()

        train_df, test_df = train_test_split(df, test_size=0.2)

        mean = torch.tensor([0.593, 0.567, 0.534])
        std = torch.tensor([0.247, 0.247, 0.247])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((600, 400)),
            transforms.Normalize(mean, std)
        ])

        train_dataset = SegmentationDataset(train_df, transform=transform)
        test_dataset = SegmentationDataset(test_df, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        return train_loader, test_loader
