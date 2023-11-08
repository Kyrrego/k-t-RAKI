'''
@ author: Yunrui Zhang

@ create time: 08/17/2023

Custom PyTorch 3D Convolution Layer to perform complex-valued convolutions. 

Reference: 
complexUtils.py (from iRAKI code: https://github.com/pdawood/iterativeRaki.git)
'''


import torch
import torch.nn as nn

from torch.nn import  Module

class complex_conv3d(Module):
    '''
    Custom PyTorch 3D Convolution Layer to perform complex-valued convolutions. 
    Basically, we change nn.Conv2d to nn.Conv3d
    '''    
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias):

        super(complex_conv3d, self).__init__()
        
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
        real_input = complex_input.real
        imag_input = complex_input.imag

        real_filter_real_input = self.real_filter(real_input)
        real_filter_imag_input = self.real_filter(imag_input)
        imag_filter_real_input = self.imag_filter(real_input)
        imag_filter_imag_input = self.imag_filter(imag_input)

        real_output = real_filter_real_input - imag_filter_imag_input
        imag_output = real_filter_imag_input + imag_filter_real_input

        complex_output = torch.complex(real_output, imag_output)

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