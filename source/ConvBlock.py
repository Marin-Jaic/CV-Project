import torch as th
from torch import nn

class ConvBlock(nn.Sequential):
    def __init__(self, kernel_size, input_features, output_features):
        super.__init__(self,
                       nn.Conv2d(kernel_size= kernel_size, 
                                 in_channels= input_features, 
                                 out_channels= output_features),
                       nn.ReLU(),
                       nn.Conv2d(kernel_size= kernel_size, 
                                 in_channels= output_features, 
                                 out_channels= output_features),
                       nn.ReLU(),
                       nn.Conv2d(kernel_size= kernel_size, 
                                 in_channels= output_features, 
                                 out_channels= output_features),
                       )