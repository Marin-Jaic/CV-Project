import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from torchvision import transforms
from .SegmentationDataset import SegmentationDataset
import torch
from torch.utils.data import DataLoader


class Preprocess:

    def get_data_loaders(self, subset=False) -> tuple:
        dataset = pq.ParquetDataset('./data')

        table = dataset.read()
        df = table.to_pandas()

        train_df, test_df = train_test_split(df, test_size=0.2)

        mean = torch.tensor([0.593, 0.567, 0.534])
        std = torch.tensor([0.247, 0.247, 0.247])

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((600, 400)),
            transforms.Normalize(mean, std)
        ])

        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((600, 400), interpolation=transforms.InterpolationMode.NEAREST)
        ])

        train_dataset = SegmentationDataset(train_df, img_transform=img_transform, mask_transform=mask_transform)
        test_dataset = SegmentationDataset(test_df, img_transform=img_transform, mask_transform=mask_transform)

        if subset:
            train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
            test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        return train_loader, test_loader
