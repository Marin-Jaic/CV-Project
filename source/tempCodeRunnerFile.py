image = image.transpose(1, 2).transpose(0, 1).unsqueeze(dim = 0)

unet = UNet(INPUT_HEIGHT= 176, INPUT_WIDTH= 176, INPUT_FEATURE_NUMBER= 1)

unet.forward(image)