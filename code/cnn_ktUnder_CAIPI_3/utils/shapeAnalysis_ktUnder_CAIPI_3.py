import numpy as np


def getPseudoKernel(num_hid_layer, layer_design):
    '''
    Function to calculate effective kernel-size in multi-layer Convolution-Networks.

    Args:
        num_hid_layer: Number of 'hidden' layers in Network. 
        layer_design: Dictionary describing Network-Architecture. Here is a example with two hidden layers:

        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                        'mask_kernel': mask_kernel, # mask_kernel is a 3D array, to specify the shape of the kernel
                            1:[256,(2,5,3)],   # the first hidden layer has 256 channels, and a kernel size of (2,5,3) in PE-, RO- and t-direction
                            2:[128,(1.1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1,1) in PE-, RO- and t-direction
                        'output_unit':[(R-1)*nC,(1,5,3)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5,3) in PE-, RO- and t-direction
                        }

    Returns:
        p_pseudo, r_pseudo, t_pseudo: effective convolution kernel dimension in PE(ky) -, RO(kx)-, and time-
        direction, respectively. 
    '''
    p_pseudo = layer_design[1][1][0]
    r_pseudo = layer_design[1][1][1]
    t_pseudo = layer_design[1][1][2]

    if num_hid_layer > 1:
        for i in range(2, num_hid_layer + 1):
            p_pseudo += layer_design[i][1][0] - 1
            r_pseudo += layer_design[i][1][1] - 1
            t_pseudo += layer_design[i][1][2] - 1

    p_pseudo += layer_design['output_unit'][1][0] - 1
    r_pseudo += layer_design['output_unit'][1][1] - 1
    t_pseudo += layer_design['output_unit'][1][2] - 1

    return p_pseudo, r_pseudo, t_pseudo


def extractDatCNN_ktUnder_CAIPI_3(acs, R, num_hid_layer, layer_design):
    '''
    Function to get Training-Data for CNNs for k-space interpolation. 
    Args: 
        acs: Auto-Calibration-Signal in shape [coil, PE, RO].
        R: Undersampling Factor.
        num_hid_layer: Number of 'hidden' layers in Network. 
        layer_design: Dictionary describing Network-Architecture.

    Returns:
        prc_data: Dictionary with keys 'src': Source-Signals.
                                        'trg': Target-Signals.
    '''
    Nk_p, Nk_r, Nk_t = getPseudoKernel(num_hid_layer, layer_design)
    print("The shape of pseudo kernels is (Nk_p, Nk_r, Nk_t): ", Nk_p, Nk_r, Nk_t)
    # mask_kernel = layer_design['mask_kernel']

    # number of coils, phase-steps, read-out-steps
    (num_coils, num_p_acs, num_r_acs, num_t_acs) = acs.shape
    print("The shape of acs is (num_coils, num_p_acs, num_r_acs, num_t_acs): ", acs.shape)

    # computing how many times the kernel fits into the block
    # in read-direction (rep_r_acs) , in phase-direction (rep_p_acs) and in time-direction (rep_t_acs)
    # 09/03: Consider the stride R in t and PE direction!
    # for acs-block
    rep_r_acs = (num_r_acs - Nk_r) + 1
    rep_t_acs = (num_t_acs - Nk_t) // R + 1
    rep_p_acs = (num_p_acs - Nk_p) // R + 1
    print("The repetitions are (for acs data): ", "rep_r_acs: ",
          rep_r_acs, "rep_t_acs: ", rep_t_acs, "rep_p_acs: ", rep_p_acs)

    # kernel_extension_p = (Nk_p - 1) * (R - 1) + Nk_p    # Dilated size of pseudo kernel in PE direction
    # 09/07: This is batch size!
    # 09/07: Set batch size to be twice of kernel size(2*R)
    # kernel_extension_p = 2 * R
    # print("Kernel extension (batch size) in PE direction is: ", kernel_extension_p)

    # determining the k-points within the kernel which are related to
    # the target points
    r_trg = Nk_r // 2
    t_trg = Nk_t // 2
    p_trg = Nk_p // 2 - 1

    # dimension of target-vector
    # 09/03: R * (R-1): number of missing points in each kernel, if diagonal undersampling is applied in ky and t direction
    dim_trg_vec = R * (R - 1) * acs.shape[0]

    # initializing the source - and target - matrices obtained from acs-block
    trg_data = np.ndarray((rep_p_acs, rep_r_acs, rep_t_acs, dim_trg_vec),
                          dtype=complex)
    # print("The shape of trg_data is (rep_p_acs, rep_r_acs, rep_t_acs, dim_trg_vec): ", trg_data.shape)

    # loop for kernel-displacement in phase-direction
    for p in range(rep_p_acs):
        # loop for kernel-displacement in read-direction
        for r in range(rep_r_acs):
            # loop for kernel-displacement in time-direction
            for t in range(rep_t_acs):
                # For R=2 ky-t undersampling specifically! We select the top right and bottom left points in each kernel directly.
                # When R=2, R(R-1)=2, so target data dimension should be 2*num_coils.
                # For R>2 ky-t undersampling, we need to flatten the kernel first and indicate the indices. (Or, simply multiply with mask_kernel!)
                trg_data[p, r, t, 0:acs.shape[0]] = acs[:, p*R,
                                                        r+r_trg, t*R+1].reshape(acs.shape[0])    # (0,1) point
                trg_data[p, r, t, acs.shape[0]:2*acs.shape[0]] = acs[:, p*R,
                                                                     r + r_trg, t*R+2].reshape(acs.shape[0])    # (0,2) point
                trg_data[p, r, t, 2*acs.shape[0]:3*acs.shape[0]] = acs[:, p*R+1,
                                                                       r+r_trg, t*R].reshape(acs.shape[0])    # (1,0) point
                trg_data[p, r, t, 3*acs.shape[0]:4*acs.shape[0]] = acs[:, p*R+1,
                                                                       r+r_trg, t*R+2].reshape(acs.shape[0])    # (1,2) point
                trg_data[p, r, t, 4*acs.shape[0]:5*acs.shape[0]] = acs[:, p*R+2,
                                                                       r+r_trg, t*R].reshape(acs.shape[0])    # (2,0) point
                trg_data[p, r, t, 5*acs.shape[0]:6*acs.shape[0]] = acs[:, p*R+2,
                                                                       r+r_trg, t*R+1].reshape(acs.shape[0])    # (2,1) point

    trg_data = np.expand_dims(trg_data, axis=1)

    # 09/03: Why src_data has this shape?
    # 10/13: Edited shape of src_data
    # src_data = np.ndarray((rep_p_acs,
    #                        num_coils,
    #                        kernel_extension_p,
    #                        num_r_acs,
    #                        num_t_acs),
    #                        dtype=complex
    #                        )
    src_data = np.ndarray((1,
                           num_coils,
                           num_p_acs,
                           num_r_acs,
                           num_t_acs),
                          dtype=complex
                          )
    # print("The shape of src_data is (rep_p_acs, num_coils, kernel_extension_p, num_r_acs, num_t_acs): ", src_data.shape)

    # for i in range(rep_p_acs):
    #     # Use all of acs data in kx and t direction, but only kernel_extension_p in ky direction.
    #     # No, more like chopping the acs data in ky direction into rep_p_acs pieces (each with kernel_extension_p points).
    #     # But why?
    #     # Because PE and t has same properties now, maybe we should do the same with t. But anyways, let's not change this for the moment.
    #     src_data[i, :, :, :, :] = acs[:, i:i+kernel_extension_p, :, :]

    # 10/13: Major changes here: we simply unsqueeze the acs data to 5D, instead of using slices and setting batches
    src_data = np.expand_dims(acs, axis=0)

    src_data = src_data.transpose((0, 2, 3, 4, 1))

    prc_data = {'src': src_data, 'trg': trg_data}

    return prc_data


def fillDatCNN_ktUnder_CAIPI_3(kspace_zf, pred_mat, R, num_hid_layer, layer_design):
    '''
    Function to re-insert estimated missing signals back into zero-filled k-space.
    Args:
        kspace_zf: Zerofilled kspace.
        pred_mat: Estimated signals.
        R: Undersampling Factor.
        num_hid_layer: Number of 'hidden' layers in Network. 
        layer_design: Dictionary describing Network-Architecture.

    Returns: 
        Reconstructed k-space in shape [coils, PE, RO].
    '''
    Nk_p, Nk_r, Nk_t = getPseudoKernel(num_hid_layer, layer_design)
    # number of coils, phase-steps, read-out-steps
    (num_coil, num_p_data, num_r_data, num_t_data) = kspace_zf.shape
    print("The shape of kspace_zf is (num_coil, num_p_data, num_r_data, num_t_data): ", kspace_zf.shape)
    print("The shape of pred_mat is: ", pred_mat.shape)

    # for data-block
    rep_r = (num_r_data - Nk_r) + 1
    rep_t = (num_t_data - Nk_t) // R + 1
    rep_p = (num_p_data - Nk_p) // R + 1
    print("The repetitions are (for kspace): ",
          "rep_r: ", rep_r, "rep_t: ", rep_t, "rep_p: ", rep_p)

    # determining the k-points within the kernel which are related to
    # the target points
    r_trg = Nk_r // 2
    t_trg = Nk_t // 2
    p_trg = Nk_p // 2 - 1

    for p in range(rep_p):
        for r in range(rep_r):
            for t in range(rep_t):
                kspace_zf[:, p*R, r+r_trg, t*R+1] = pred_mat[p,
                                                             r, t, 0:num_coil].reshape((num_coil,))    # (0,1) point
                kspace_zf[:, p*R, r+r_trg, t*R+2] = pred_mat[p,
                                                             r, t, num_coil:2*num_coil].reshape((num_coil,))    # (0,2) point
                kspace_zf[:, p*R+1, r+r_trg, t*R] = pred_mat[p,
                                                             r, t, 2*num_coil:3*num_coil].reshape((num_coil,))    # (1,0) point
                kspace_zf[:, p*R+1, r+r_trg, t*R+2] = pred_mat[p,
                                                               r, t, 3*num_coil:4*num_coil].reshape((num_coil,))    # (1,2) point
                kspace_zf[:, p*R+2, r+r_trg, t*R] = pred_mat[p,
                                                             r, t, 4*num_coil:5*num_coil].reshape((num_coil,))    # (2,0) point
                kspace_zf[:, p*R+2, r+r_trg, t*R+1] = pred_mat[p,
                                                               r, t, 5*num_coil:6*num_coil].reshape((num_coil,))    # (2,1) point

    return kspace_zf
