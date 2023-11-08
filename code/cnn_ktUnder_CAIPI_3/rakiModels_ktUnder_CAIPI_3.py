#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:43:46 2022
@author: ubuntu
"""
from .utils.shapeAnalysis_ktUnder_CAIPI_3 import extractDatCNN_ktUnder_CAIPI_3, fillDatCNN_ktUnder_CAIPI_3
from .utils.complexCNN_ktUnder_CAIPI_3 import complexNet_ktUnder_CAIPI_3
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
'''
import sys
sys.path.append('./utils/')

from importlib import reload
import complexCNN_ktUnder
reload(complexCNN_ktUnder)
'''


# 08/30: fine-tune the learning rate. Default is 0.0005.
RAKI_RECO_DEFAULT_LR = 0.0005
RAKI_RECO_DEFAULT_EPOCHS = 500

IRAKI_RECO_DEFAULT_INIT_LR = 5e-4
IRAKI_RECO_DEFAULT_LR_DECAY = {
    4: 3e-5,
    5: 4e-5,
    6: 6e-5
}
IRAKI_RECO_DEFAULT_ACS_NUM = 65


def rakiReco_ktUnder_CAIPI_3(kspc_zf, acs, R, layer_design):
    '''
    This function trains RAKI, and puts the interpolated signals 
    into zero-filled k-space.

    Args:
        kspc_zf: Zero-filled k-space, not including acs, in shape [coils, PE, RO].
        acs: Auto-Calibration-Signal, in shape [coils, PE, RO].
        R: Undersampling-Factor.
        layer_design: Network-Architecture. Here is a example with two hidden layers:

        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                        'mask_kernel': mask_kernel, # mask_kernel is a 3D array, to specify the shape of the kernel
                            1:[256,(2,5,3)],   # the first hidden layer has 256 channels, and a kernel size of (2,5,3) in PE-, RO- and t-direction
                            2:[128,(1.1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1,1) in PE-, RO- and t-direction
                        'output_unit':[(R-1)*nC,(1,5,3)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5,3) in PE-, RO- and t-direction
                        }

    Returns:
        network_reco: RAKI k-space reconstruction, in shape [coils, PE, RO].
    '''
    print('Starting Standard RAKI (in k-t undersampling)...')
    # Get Source- and Target Signals
    prc_data = extractDatCNN_ktUnder_CAIPI_3(acs,
                                             R=R,
                                             num_hid_layer=layer_design['num_hid_layer'],
                                             layer_design=layer_design)

    trg_kspc = prc_data['trg'].transpose((0, 4, 1, 2, 3))
    src_kspc = prc_data['src'].transpose((0, 4, 1, 2, 3))
    trg_kspc = trg_kspc.transpose((2, 1, 0, 3, 4))

    src_kspc = torch.from_numpy(src_kspc).type(torch.complex64)
    trg_kspc = torch.from_numpy(trg_kspc).type(torch.complex64)

    net = complexNet_ktUnder_CAIPI_3(layer_design, R=R)
    criterion = nn.MSELoss()

    # print the names of the parameters of the network (net.parameters()), however, the tensors are a bit too long
    # print("The parameters of the network are: ", list(net.parameters()))

    optimizer = optim.Adam(net.parameters(), lr=RAKI_RECO_DEFAULT_LR)

    for _ in trange(RAKI_RECO_DEFAULT_EPOCHS):    # trange: show progress bar
        # print("the parameters of the network are: ", net.conv1.kernel_real[35][5])
        optimizer.zero_grad()
        # print("The shape of src_kspc is: ", src_kspc.shape)
        # print("The shape of trg_kspc is: ", trg_kspc.shape)

        pred_kspc = net(src_kspc)['tot']
        # print("The shape of pred_kspc is: ", pred_kspc.shape)

        loss = (criterion(pred_kspc.real, trg_kspc.real)
                + criterion(pred_kspc.imag, trg_kspc.imag))
        loss.backward()
        optimizer.step()

        '''# print the gradient of the weights, see if they're changing
        print("The gradients of the weights are: ", net.conv1.real_filter.weight.grad[35][5])
        print("The weights are: ", net.conv1.real_filter.weight[35][5])
        print("The mask_kernel is: ", net.conv1.mask_kernel[35][5])'''

    kspc_zf_input = np.expand_dims(kspc_zf, axis=0)
    kspc_zf_input = torch.from_numpy(kspc_zf_input).type(torch.complex64)
    # Estimate missing signals
    kspc_pred = net(kspc_zf_input)['tot']
    kspc_pred = kspc_pred.detach().numpy()
    kspc_pred = kspc_pred.transpose((0, 2, 3, 4, 1))
    kspc_pred = np.squeeze(kspc_pred)

    # Put estimated signals bach into zero-filled kspace
    network_reco = fillDatCNN_ktUnder_CAIPI_3(kspc_zf,
                                              kspc_pred,
                                              R,
                                              num_hid_layer=layer_design['num_hid_layer'],
                                              layer_design=layer_design)
    print('Finished Standard RAKI...')
    return network_reco
