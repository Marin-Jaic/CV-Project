import torch

from source.data_prep.preprocess import Preprocess
from source.models.UnetV2 import UNetV2
from training import train_model, test_model
from matplotlib  import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 10
subset_size = 6000
num_classes = 18

# A subsample of 1k elements (by default) from the dataset is taken
train_loader, test_loader = Preprocess().get_data_loaders(subset=True, mask_classes=num_classes,
                                                          subset_size=subset_size)

unet = UNetV2(n_channels=3, n_classes=3).to(device)

train_losses, epoch_iou = train_model(model=unet, train_loader=train_loader, epochs=epochs)
test_losses, test_iou = test_model(model=unet, test_loader=test_loader)

plt.plot(range(epochs), train_losses, label='Training loss')
plt.title('Training Loss - 3 Class Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(range(epochs), torch.tensor(epoch_iou).cpu(), label='Training IoU')
plt.title('Training IoU - 3 Class Model')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.show()

# Save the model weights
PATH = f"trained_models/unet_{epochs}_{subset_size}_{num_classes}.pt"
torch.save(unet.state_dict(), PATH)
