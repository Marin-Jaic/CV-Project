import torch as th
from ConvBlock import ConvBlock
from torch import nn

CONVOLUTION_KERNEL_SIZE = 3
DECONVOLUTION_KERNEL_SIZE = 2
POOL_KERNEL_SIZE = 2
STARTING_FEATURE_NUMBER = 1
FEATURE_NUMBER = 64
OUTPUT_FEATURE_NUMBER = 2

class UNet(nn.Module):
    def __init__(self, 
                 CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                 DECONVOLUTION_KERNEL_SIZE = DECONVOLUTION_KERNEL_SIZE, 
                 POOL_KERNEL_SIZE = POOL_KERNEL_SIZE, 
                 STARTING_FEATURE_NUMBER = STARTING_FEATURE_NUMBER, 
                 FEATURE_NUMBER = FEATURE_NUMBER,
                 OUTPUT_FEATURE_NUMBER = OUTPUT_FEATURE_NUMBER):
        super.__init__(self)

        self.pool = nn.MaxPool2d()
        
        self.first_contracting_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                                      input_features = STARTING_FEATURE_NUMBER, 
                                                      output_features = FEATURE_NUMBER)

        self.second_contracting_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                                       input_features = FEATURE_NUMBER, 
                                                       output_features = FEATURE_NUMBER * 2)
        
        self.third_contracting_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                                      input_features = FEATURE_NUMBER * 2, 
                                                      output_features = FEATURE_NUMBER * 4)
        
        self.fourth_contracting_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                                       input_features = FEATURE_NUMBER * 4, 
                                                       output_features = FEATURE_NUMBER * 8)
        
        self.bridge = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                input_features = FEATURE_NUMBER * 8, 
                                output_features = FEATURE_NUMBER * 16) 
        
        self.first_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 16,
                                                      out_channels = FEATURE_NUMBER * 8)
        
        self.first_expansive_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 16, 
                                              output_features = FEATURE_NUMBER * 8)
        
        self.second_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 8,
                                                      out_channels = FEATURE_NUMBER * 4)
        
        self.second_expansive_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 8, 
                                              output_features = FEATURE_NUMBER * 4)
        
        self.third_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 4,
                                                      out_channels = FEATURE_NUMBER * 2)
        
        self.third_expansive_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 4, 
                                              output_features = FEATURE_NUMBER * 2)
        
        self.fourth_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 2,
                                                      out_channels = FEATURE_NUMBER)
        
        self.fourth_expansive_conv = ConvBlock(CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 2, 
                                              output_features = FEATURE_NUMBER)
        
        self.final_convolution = nn.Conv2d(kernel_size = 1,
                                           in_channels = FEATURE_NUMBER,
                                           out_channels = OUTPUT_FEATURE_NUMBER)

    def forward(self, x):
        #I would imagine the input is in dimensions [batch_size, channels, height, width]
        contracting_first_feature_map = self.first_contracting_conv(x)
        contracting_second_feature_map = self.second_contracting_conv(self.pool(contracting_first_feature_map))
        contracting_third_feature_map = self.third_contracting_conv(self.pool(contracting_second_feature_map))
        contracting_fourth_feature_map = self.fourth_contracting_conv(self.pool(contracting_third_feature_map))

        #TODO CROPPING
        expansive_first_feature_map = self.bridge(self.pool(contracting_fourth_feature_map))
        concatenated = th.cat(contracting_fourth_feature_map,
                            self.first_deconvolution(expansive_first_feature_map),
                            dim = 1)
        
        expansive_second_feature_map = self.first_expansive_conv(concatenated)
        concatenated = th.cat(contracting_third_feature_map,
                              self.first_deconvolution(expansive_second_feature_map),
                              dim = 1)
        
        expansive_third_feature_map = self.second_expansive_conv(concatenated)
        concatenated = th.cat(contracting_second_feature_map,
                              self.first_deconvolution(expansive_third_feature_map),
                              dim = 1)
        
        expansive_fourth_feature_map = self.third_expansive_conv(concatenated)
        concatenated = th.cat(contracting_first_feature_map,
                              self.first_deconvolution(expansive_fourth_feature_map),
                              dim = 1)
        
        expansive_fourth_feature_map = self.fourth_expansive_conv(concatenated)

        output = self.final_convolution(expansive_fourth_feature_map)

        return output
    



        