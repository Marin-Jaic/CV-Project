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

        assert INPUT_HEIGHT % 2 == 0, "height should be an even number"
        assert INPUT_WIDTH % 2 == 0, "width should be an even number" 
        
        self.input_height = INPUT_HEIGHT
        self.input_width = INPUT_WIDTH
        self.convolution_kernel_size = CONVOLUTION_KERNEL_SIZE
        self.pool_kernel_size = POOL_KERNEL_SIZE
        self.deconvolution_kernel_size = DECONVOLUTION_KERNEL_SIZE
        self.deconvolution_stride = DECONVOLUTION_STRIDE
        
        #a list containing tuples of cropped feature map dimensions in the form of (height, width) 
        #in order their concatenatation during the pipeline
        self.cropped_dimensions = self.expansive_cropped_dimensions(self.bridge_dimensions())
        
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
    
    def bridge_dimensions(self):
        height = self.input_height
        width = self.input_width
        kernel = self.convolution_kernel_size

        #for standard depth 4
        for i in range(4):
            height = (height - 3 * kernel + 3) / 2
            width = (width - 3 * kernel + 3) / 2
        
        return (height, width)

    def crop_feature_map(self, cropped_dimensions, feature_map):
        cropped_height = cropped_dimensions[0]
        cropped_width = cropped_dimensions[1]

        feature_map_height = feature_map.shape[2]
        feature_map_width = feature_map.shape[3]

        height_first_index = (feature_map_height - cropped_height) / 2
        width_first_index = (feature_map_width - cropped_width) / 2

        return feature_map[:, :, height_first_index:height_first_index + cropped_height, width_first_index:width_first_index + cropped_width]

    def expansive_cropped_dimensions(self, bridge_dimensions):
        height = bridge_dimensions[0]
        width = bridge_dimensions[1]
        kernel = self.convolution_kernel_size

        dimensions = []

        for i in range(4):
            #formulas incomplete but since we use the default dilation and no padding, it suffices
            height = (height - 1) * self.deconvolution_stride + self.deconvolution_kernel_size 
            width = (width - 1) * self.deconvolution_stride + self.deconvolution_kernel_size 

            dimensions.append((height, width))

            height = height - 3 * kernel + 3
            width = width - 3 * kernel + 3
        
        return dimensions

    def forward(self, x):
        #I would imagine the input is in dimensions [batch_size, channels, height, width]
        contracting_first_feature_map = self.first_contracting_conv(x)
        contracting_second_feature_map = self.second_contracting_conv(self.pool(contracting_first_feature_map))
        contracting_third_feature_map = self.third_contracting_conv(self.pool(contracting_second_feature_map))
        contracting_fourth_feature_map = self.fourth_contracting_conv(self.pool(contracting_third_feature_map))

        
        expansive_first_feature_map = self.bridge(self.pool(contracting_fourth_feature_map))
        concatenated = th.cat((self.crop_feature_map(self.cropped_dimensions[0], contracting_fourth_feature_map),
                            self.first_deconvolution(expansive_first_feature_map)),
                            dim = 1)
        
        expansive_second_feature_map = self.first_expansive_conv(concatenated)
        concatenated = th.cat((self.crop_feature_map(self.cropped_dimensions[1], contracting_third_feature_map),
                              self.first_deconvolution(expansive_second_feature_map)),
                              dim = 1)
        
        expansive_third_feature_map = self.second_expansive_conv(concatenated)
        concatenated = th.cat((self.crop_feature_map(self.cropped_dimensions[2], contracting_second_feature_map),
                              self.first_deconvolution(expansive_third_feature_map)),
                              dim = 1)
        
        expansive_fourth_feature_map = self.third_expansive_conv(concatenated)
        concatenated = th.cat((self.crop_feature_map(self.cropped_dimensions[3],contracting_first_feature_map),
                              self.first_deconvolution(expansive_fourth_feature_map)),
                              dim = 1)
        
        expansive_fourth_feature_map = self.fourth_expansive_conv(concatenated)

        output = self.final_convolution(expansive_fourth_feature_map)

        return output
    



        