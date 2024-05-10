from PIL import Image
import torch as th
import numpy as np
from UNet import UNet

image = Image.open("test_image_big.png")
image = np.array(image)
image = th.Tensor(image)

image = image.unsqueeze(dim = 0).unsqueeze(dim = 0)
print(image.shape)
#image = image.transpose(1, 2).transpose(0, 1).unsqueeze(dim = 0)

unet = UNet(INPUT_HEIGHT= 176, INPUT_WIDTH= 176, INPUT_FEATURE_NUMBER= 1)

unet.forward(image)
