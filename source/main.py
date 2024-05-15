from PIL import Image
import torch as th
import numpy as np
from UNet import UNet
from data_prep.preprocess import Preprocess

# image = Image.open("test_image_big.png")
# image = np.array(image)
# image = th.Tensor(image)

# image = image.unsqueeze(dim = 0).unsqueeze(dim = 0)

# pool = th.nn.MaxPool2d(kernel_size=2, stride=2)
# print(pool(image).shape)

train_loader, test_loader = Preprocess().get_data_loaders()

# image = image.transpose(1, 2).transpose(0, 1).unsqueeze(dim = 0)
image = zeros_tensor = th.zeros(1, 1, 512, 512)
unet = UNet(INPUT_HEIGHT= 512, INPUT_WIDTH= 512, INPUT_FEATURE_NUMBER= 1)

unet(image)
