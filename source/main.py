import torch

from source.data_prep.preprocess import Preprocess
from source.models.UNet import UNet
from source.models.UnetV2 import UNetV2
from training import train_model, test_model
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# A subsample of 1k elements from the dataset is taken
train_loader, test_loader = Preprocess().get_data_loaders(subset=True)

epochs = 10

# The Unet model produces output masks with dim (404, 212), idk whats wrong
unet = UNet(INPUT_HEIGHT=512, INPUT_WIDTH=522, INPUT_FEATURE_NUMBER=3)
unet = unet.to(device)

unet_v2 = UNetV2(n_channels=3, n_classes=3)
unet_v2 = unet_v2.to(device)

train_losses, epoch_iou = train_model(model=unet_v2, train_loader=train_loader, epochs=epochs)
test_losses, test_iou = test_model(model=unet_v2, test_loader=test_loader)

plt.plot(train_losses, range(epochs), label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(epoch_iou.cpu(), range(epochs), label='Training IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.show()
