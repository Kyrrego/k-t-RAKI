'''
@ author: Yunrui Zhang

@ create time: 08/17/2023

Custom PyTorch 3D Convolution Layer to perform complex-valued convolutions. 

Reference: 
complexCNN.py (from iRAKI code: https://github.com/pdawood/iterativeRaki.git)
'''

from .complexUtils_3D import cLeakyReLu
from .complexUtils_ktUnder_CAIPI_3 import complex_conv_ktUnder_CAIPI_3
from .complexUtils_3D import complex_conv3d
import torch.nn as nn

#from .complexUtils import complex_conv2d
'''
from importlib import reload
import complexUtils_ktUnder
reload(complexUtils_ktUnder)
import complexUtils_3D
reload(complexUtils_3D)
'''


class complexNet_ktUnder_CAIPI_3(nn.Module):
    """Torch module for creation of complex neural network with CNN arch.

       Attributes
        R : int 
        undersampling factor 

        layer_design: dict
        Network-Architecture. The size of the kernel is : [channels, (y,x,t)]
        Here is a example with two hidden layers:

        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                        'mask_kernel': mask_kernel, # mask_kernel is a 3D array, to specify the shape of the kernel
                            1:[256,(2,5,3)],   # the first hidden layer has 256 channels, and a kernel size of (2,5,3) in PE-, RO- and t-direction
                            2:[128,(1.1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1,1) in PE-, RO- and t-direction
                        'output_unit':[(R-1)*nC,(1,5,3)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5,3) in PE-, RO- and t-direction
                        }


    """

    def __init__(self, layer_design, R):
        super(complexNet_ktUnder_CAIPI_3, self).__init__()
        # input layer: use customized kernel
        self.conv1 = complex_conv_ktUnder_CAIPI_3(in_channels=layer_design['input_unit'],
                                                  out_channels=layer_design[1][0],
                                                  kernel_size=layer_design[1][1],
                                                  stride=[R, 1, R],
                                                  padding=0,
                                                  dilation=[1, 1, 1],
                                                  groups=1,
                                                  bias=False,
                                                  mask_kernel=layer_design[1][2])
        self.cnt = 1
        # hidden layer
        for k in range(layer_design['num_hid_layer'] - 1):
            name = 'conv' + str(k + 2)
            setattr(
                self, name,
                complex_conv3d(in_channels=layer_design[k+1][0],
                               out_channels=layer_design[k+2][0],
                               kernel_size=layer_design[k+2][1],
                               stride=1,
                               padding=0,
                               dilation=[1, 1, 1],
                               groups=1,
                               bias=False)
            )
            self.cnt += 1

        # output layer
        name = 'conv' + str(layer_design['num_hid_layer'] + 1)
        setattr(
            self, name,
            complex_conv3d(in_channels=layer_design[self.cnt][0],
                           out_channels=layer_design['output_unit'][0],
                           kernel_size=layer_design['output_unit'][1],
                           stride=1,
                           padding=0,
                           dilation=[1, 1, 1],
                           groups=1,
                           bias=False)
        )
        self.cnt += 1

    def forward(self, input_data):
        # input layer
        x = self.conv1(input_data)
        x = cLeakyReLu(x)
        # hidden layer
        for numLayer in range(self.cnt - 2):
            convLayer = getattr(self, 'conv' + str(numLayer + 2))
            x = convLayer(x)
            x = cLeakyReLu(x)
        # output layer, without ReLU
        convLayer = getattr(self, 'conv' + str(self.cnt))
        x = convLayer(x)
        conv = {'tot': x}
        return conv
