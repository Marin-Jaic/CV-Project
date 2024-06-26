import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from torchvision import transforms
from .SegmentationDataset import SegmentationDataset
import torch
from torch.utils.data import DataLoader


class Preprocess:

    def get_data_loaders(self, mask_classes, subset=False, use_simple_mask=True, batch_size=4, subset_size=1000):
        dataset = pq.ParquetDataset('data')

        table = dataset.read()
        df = table.to_pandas()

        train_df, test_val_df = train_test_split(df, test_size=0.2)
        test_df, val_df = train_test_split(test_val_df, test_size=0.5)

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

        train_dataset = SegmentationDataset(train_df,
                                            img_transform=img_transform,
                                            mask_transform=mask_transform,
                                            mask_classes=mask_classes,
                                            use_simple_mask=use_simple_mask)
        val_dataset = SegmentationDataset(test_df,
                                          img_transform=img_transform,
                                          mask_transform=mask_transform,
                                          mask_classes=mask_classes,
                                          use_simple_mask=use_simple_mask)
        test_dataset = SegmentationDataset(test_df,
                                           img_transform=img_transform,
                                           mask_transform=mask_transform,
                                           mask_classes=mask_classes,
                                           use_simple_mask=use_simple_mask)

        if subset:
            train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
