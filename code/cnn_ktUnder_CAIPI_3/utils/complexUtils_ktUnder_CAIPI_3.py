'''
@ author: Yunrui Zhang

@ create time: 08/17/2023

Custom PyTorch 3D Convolution Layer to perform complex-valued convolutions. 

Reference: 
complexUtils.py (from iRAKI code: https://github.com/pdawood/iterativeRaki.git)
'''


import torch
import torch.nn as nn

from torch.nn import Module


class complex_conv_ktUnder_CAIPI_3(Module):
    '''
    Custom PyTorch 3D Convolution Layer to perform complex-valued convolutions. 
    Basically, we change nn.Conv2d to nn.Conv3d

    On 08/22: 
    Define a new parameter: mask_kernel. To specify the shape of the kernel. 
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, mask_kernel):

        super(complex_conv_ktUnder_CAIPI_3, self).__init__()

        mask_kernel = nn.Parameter(torch.tensor(
            mask_kernel, dtype=torch.float32))
        self.mask_kernel = nn.Parameter(
            mask_kernel, requires_grad=True)    # True or false?

        ''' 08/30 afternoon: Maybe try to discard the kernels and use weight directly? 
        #Init self.kernel: nparray of size [out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]]
        self.kernel_real = nn.Parameter(torch.ones(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]), requires_grad=True)
        self.kernel_imag = nn.Parameter(torch.ones(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]), requires_grad=True)'''

        self.real_filter = nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups,
                                     bias)

        self.imag_filter = nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups,
                                     bias)

    def forward(self, complex_input):

        # multiply the kernel with the mask in every forward, so certain spots are zeros
        # print("self.kernel_real is: ", self.kernel_real[35][5][:,0,:])
        self.real_filter.weight.data = self.real_filter.weight.data * self.mask_kernel
        self.imag_filter.weight.data = self.imag_filter.weight.data * self.mask_kernel

        # Define the weight manually.
        ''' self.real_filter.weight = nn.Parameter(custom_kernel_real, requires_grad=True)
        self.imag_filter.weight = nn.Parameter(custom_kernel_imag, requires_grad=True)'''

        '''# Copy the weight to self.kernel_real and self.kernel_imag.
        # Not sure if I should write it again here. 
        self.kernel_real = self.real_filter.weight
        self.kernel_imag = self.imag_filter.weight'''

        real_input = complex_input.real
        imag_input = complex_input.imag

        ''' 
        print("The shape of real_input is: ", real_input.shape)
        print("The shape of imag_input is: ", imag_input.shape)
        print("The shape of real_filter.weight is: ", self.real_filter.weight.shape)
        print("The shape of imag_filter.weight: ", self.imag_filter.weight.shape)
        print("The shape of real_filter.stride is : ", self.real_filter.stride) 
        print("The shape of custom_kernel is: ", custom_kernel_real.shape)
        # print("The properties of real_filter is ", self.real_filter.__dict__)'''

        real_filter_real_input = self.real_filter(real_input)
        real_filter_imag_input = self.real_filter(imag_input)
        imag_filter_real_input = self.imag_filter(real_input)
        imag_filter_imag_input = self.imag_filter(imag_input)

        real_output = real_filter_real_input - imag_filter_imag_input
        imag_output = real_filter_imag_input + imag_filter_real_input

        complex_output = torch.complex(real_output, imag_output)

        '''Print the weights and the corresponding kernel_real. However, this can be achieved by printing in every epoch, so we just write it in rakiModels_ktUnder.py
        # print("real_filter.weight is: ", self.real_filter.weight[35][5][:,0,:])
        # print("self.kernel_real is: (after training) ", self.kernel_real[35][5][:,0,:])'''

        return complex_output


def cLeakyReLu(complex_input):
    '''
    Custom activation function based on Leaky Rectifier Unit to 
    perform complex-valued convolutions. 
    '''
    negative_slope = 0.5
    real_input = complex_input.real
    imag_input = complex_input.imag
    m = nn.LeakyReLU(negative_slope)

    m_real = m(real_input)
    m_imag = m(imag_input)

    complex_output = torch.complex(m_real, m_imag)

    return complex_output
