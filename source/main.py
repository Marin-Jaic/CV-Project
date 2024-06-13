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
subset_size = 6000
num_classes = 3

# A subsample of 1k elements (by default) from the dataset is taken
train_loader, val_loader, test_loader = Preprocess().get_data_loaders(subset=True, mask_classes=num_classes,
                                                                      subset_size=subset_size)

unet = UNetV2(n_channels=3, n_classes=3).to(device)
# unet.load_state_dict(torch.load("trained_models/unet_10_6000_3.pt"))

train_losses, train_iou, val_loss, val_iou = train_model(model=unet, train_loader=train_loader,
                                                         val_loader=val_loader, epochs=epochs)
test_losses, test_iou = test_model(model=unet, test_loader=test_loader)

plt.plot(range(epochs), train_losses, label='Training loss')
plt.title(f'Training Loss - {num_classes} Class Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(range(epochs), torch.tensor(train_iou).cpu(), label='Training IoU')
plt.title(f'Training IoU - {num_classes} Class Model')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.show()

plt.plot(range(epochs), train_losses, label='Validation loss')
plt.title(f'Training Loss - {num_classes} Class Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(range(epochs), torch.tensor(val_iou).cpu(), label='Validation IoU')
plt.title(f'Training IoU - {num_classes} Class Model')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.show()
# Test Loss: ~0.601, Test IoU: ~0.916 for 18 classes
# Test Loss: 0.6006672978401184, Test IoU: 0.9173608422279358 for 3 classes

# Save the model weights
PATH = f"trained_models/unet_{epochs}_{subset_size}_{num_classes}.pt"
torch.save(unet.state_dict(), PATH)
