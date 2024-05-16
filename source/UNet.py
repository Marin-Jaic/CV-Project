import torch as th
from ConvBlock import ConvBlock
from torch import nn

CONVOLUTION_KERNEL_SIZE = 3
DECONVOLUTION_KERNEL_SIZE = 2
POOL_KERNEL_SIZE = 2
INPUT_FEATURE_NUMBER = 1
FEATURE_NUMBER = 64
OUTPUT_FEATURE_NUMBER = 2

class UNet(nn.Module):
    def __init__(self, 
                 INPUT_HEIGHT,
                 INPUT_WIDTH,
                 CONVOLUTION_KERNEL_SIZE = CONVOLUTION_KERNEL_SIZE, 
                 DECONVOLUTION_KERNEL_SIZE = DECONVOLUTION_KERNEL_SIZE, 
                 POOL_KERNEL_SIZE = POOL_KERNEL_SIZE, 
                 POOL_STRIDE = 2,
                 INPUT_FEATURE_NUMBER = INPUT_FEATURE_NUMBER, 
                 FEATURE_NUMBER = FEATURE_NUMBER,
                 OUTPUT_FEATURE_NUMBER = OUTPUT_FEATURE_NUMBER,
                 DECONVOLUTION_STRIDE = 2):
        super(UNet, self).__init__()

        #assert INPUT_HEIGHT % 2 == 0, "height should be an even number"
        #assert INPUT_WIDTH % 2 == 0, "width should be an even number" 
        
        self.input_height = INPUT_HEIGHT
        self.input_width = INPUT_WIDTH
        self.convolution_kernel_size = CONVOLUTION_KERNEL_SIZE
        self.pool_kernel_size = POOL_KERNEL_SIZE
        self.pool_stride = POOL_STRIDE
        self.deconvolution_kernel_size = DECONVOLUTION_KERNEL_SIZE
        self.deconvolution_stride = DECONVOLUTION_STRIDE

        #MAYBE ITS AVERAGE POOL
        self.pool = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride = POOL_STRIDE)
        
        self.first_contracting_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                                      input_features = INPUT_FEATURE_NUMBER, 
                                                      output_features = FEATURE_NUMBER)

        self.second_contracting_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                                       input_features = FEATURE_NUMBER, 
                                                       output_features = FEATURE_NUMBER * 2)
        
        self.third_contracting_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                                      input_features = FEATURE_NUMBER * 2, 
                                                      output_features = FEATURE_NUMBER * 4)
        
        self.fourth_contracting_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                                       input_features = FEATURE_NUMBER * 4, 
                                                       output_features = FEATURE_NUMBER * 8)
        
        self.bridge = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                input_features = FEATURE_NUMBER * 8, 
                                output_features = FEATURE_NUMBER * 16) 
        
        self.first_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 16,
                                                      out_channels = FEATURE_NUMBER * 8,
                                                      stride = DECONVOLUTION_STRIDE) 
        
        self.first_expansive_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 16, 
                                              output_features = FEATURE_NUMBER * 8)
        
        self.second_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 8,
                                                      out_channels = FEATURE_NUMBER * 4,
                                                      stride = DECONVOLUTION_STRIDE)
        
        self.second_expansive_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 8, 
                                              output_features = FEATURE_NUMBER * 4)
        
        self.third_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 4,
                                                      out_channels = FEATURE_NUMBER * 2,
                                                      stride = DECONVOLUTION_STRIDE)
        
        self.third_expansive_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 4, 
                                              output_features = FEATURE_NUMBER * 2)
        
        self.fourth_deconvolution = nn.ConvTranspose2d(kernel_size = DECONVOLUTION_KERNEL_SIZE,
                                                      in_channels = FEATURE_NUMBER * 2,
                                                      out_channels = FEATURE_NUMBER,
                                                      stride = DECONVOLUTION_STRIDE)
        
        self.fourth_expansive_conv = ConvBlock(kernel_size = CONVOLUTION_KERNEL_SIZE, 
                                              input_features = FEATURE_NUMBER * 2, 
                                              output_features = FEATURE_NUMBER)
        
        self.final_convolution = nn.Conv2d(kernel_size = 1,
                                           in_channels = FEATURE_NUMBER,
                                           out_channels = OUTPUT_FEATURE_NUMBER)

        self.apply(self.weight_initialization)

    def weight_initalization(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            N = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            std_dev = (2.0 / N) ** 0.5
            nn.init.normal_(layer.weight, mean=0, std=std_dev)

            if layer.bias is not None:
                #might wanna change this
                nn.init.constant_(layer.bias, 0)

    def crop_feature_map(self, cropped_height, cropped_width, feature_map):
        feature_map_height = feature_map.shape[2]
        feature_map_width = feature_map.shape[3]

        height_first_index = round((feature_map_height - cropped_height) / 2)
        width_first_index = round((feature_map_width - cropped_width) / 2)

        return feature_map[:, :, height_first_index:height_first_index + cropped_height, width_first_index:width_first_index + cropped_width]

    def forward(self, x):
        #I would imagine the input is in dimensions [batch_size, channels, height, width]
        contracting_first_feature_map = self.first_contracting_conv(x)
        #print(contracting_first_feature_map.shape)
        contracting_second_feature_map = self.second_contracting_conv(self.pool(contracting_first_feature_map))
        #print(contracting_second_feature_map.shape)
        contracting_third_feature_map = self.third_contracting_conv(self.pool(contracting_second_feature_map))
        #print(contracting_third_feature_map.shape)
        contracting_fourth_feature_map = self.fourth_contracting_conv(self.pool(contracting_third_feature_map))
        #print(contracting_fourth_feature_map.shape)
        
        bridge_feature_map = self.first_deconvolution(self.bridge(self.pool(contracting_fourth_feature_map)))
        concatenated = th.cat((self.crop_feature_map(bridge_feature_map.shape[2], 
                                                     bridge_feature_map.shape[3], 
                                                     contracting_fourth_feature_map),
                            bridge_feature_map),
                            dim = 1)
        
        expansive_first_feature_map = self.second_deconvolution(self.first_expansive_conv(concatenated))
        concatenated = th.cat((self.crop_feature_map(expansive_first_feature_map.shape[2],
                                                     expansive_first_feature_map.shape[3],
                                                     contracting_third_feature_map),
                              expansive_first_feature_map),
                              dim = 1)
        
        expansive_second_feature_map = self.third_deconvolution(self.second_expansive_conv(concatenated))
        concatenated = th.cat((self.crop_feature_map(expansive_second_feature_map.shape[2],
                                                     expansive_second_feature_map.shape[3],
                                                     contracting_second_feature_map),
                              expansive_second_feature_map),
                              dim = 1)
        
        expansive_third_feature_map = self.fourth_deconvolution(self.third_expansive_conv(concatenated))
        concatenated = th.cat((self.crop_feature_map(expansive_third_feature_map.shape[2],
                                                     expansive_third_feature_map.shape[3],
                                                     contracting_first_feature_map),
                              expansive_third_feature_map),
                              dim = 1)
        
        expansive_fourth_feature_map = self.fourth_expansive_conv(concatenated)

        output = self.final_convolution(expansive_fourth_feature_map)

        return output
    



        