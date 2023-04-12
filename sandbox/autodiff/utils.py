'''
HISTORY
    - 2022/01/02 (GOX): copied from ML_WFS private github repo
'''
import numpy as np
# import math
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftshift, ifftshift
import aotools
from simulator_with_vortex import Vortex, crop_array

# def fft2(tensor_re, tensor_im, shift=False):
#     """Applies a 2D fft to the complex tensor represented by tensor_re and _im"""
#     # fft2
#     (tensor_out_re, tensor_out_im) = torch.fft(
#         torch.stack((tensor_re, tensor_im), 4), 2, True).split(1, 4)
#
#     tensor_out_re = tensor_out_re.squeeze(4)
#     tensor_out_im = tensor_out_im.squeeze(4)
#
#     # apply fftshift
#     if shift:
#         tensor_out_re = fftshift(tensor_out_re)
#         tensor_out_im = fftshift(tensor_out_im)
#
#     return tensor_out_re, tensor_out_im
#
#
# def ifftshift(tensor):
#     """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]
#
#     shifts the width and heights
#     """
#     size = tensor.size()
#     tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
#     tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
#     return tensor_shifted
#
#
# def fftshift(tensor):
#     """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]
#
#     shifts the width and heights
#     """
#     size = tensor.size()
#     tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
#     tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
#     return tensor_shifted
#
#
# def ifft2(tensor_re, tensor_im, shift=False):
#     """Applies a 2D ifft to the complex tensor represented by tensor_re and _im"""
#     tensor_out = torch.stack((tensor_re, tensor_im), 4)
#
#     if shift:
#         tensor_out = ifftshift(tensor_out)
#     (tensor_out_re, tensor_out_im) = torch.ifft(tensor_out, 2, True).split(1, 4)
#
#     tensor_out_re = tensor_out_re.squeeze(4)
#     tensor_out_im = tensor_out_im.squeeze(4)
#
#     return tensor_out_re, tensor_out_im

# def roll_torch(tensor, shift, axis):
#     """implements numpy roll() or Matlab circshift() functions for tensors"""
#     if shift == 0:
#         return tensor
#
#     if axis < 0:
#         axis += tensor.dim()
#
#     dim_size = tensor.size(axis)
#     after_start = dim_size - shift
#     if shift < 0:
#         after_start = -shift
#         shift = dim_size - abs(shift)
#
#     before = tensor.narrow(axis, 0, dim_size - shift)
#     after = tensor.narrow(axis, after_start, shift)
#     return torch.cat([after, before], axis)


def pad_stacked_complex(field, pad_width, padval=0, mode='constant'):
    """Helper for pad_image() that pads a real padval in a complex-aware manner"""
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width, mode=mode)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, mode=mode, value=padval)
        imag = nn.functional.pad(imag, pad_width, mode=mode, value=0)
        return torch.stack((real, imag), -1)


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field


def combine_zernike_basis(coeffs, basis, return_phase=False):
    """
    Multiplies the Zernike coefficients and basis functions while preserving
    dimensions

    :param coeffs: torch tensor with coeffs, see propagation_ASM_zernike
    :param basis: the output of compute_zernike_basis, must be same length as coeffs
    :param return_phase:
    :return: A Complex64 tensor that combines coeffs and basis.
    """

    if len(coeffs.shape) < 3:
        coeffs = torch.reshape(coeffs, (coeffs.shape[0], 1, 1))

    # combine zernike basis and coefficients
    zernike = (coeffs * basis).sum(0, keepdim=True)

    # shape to [1, len(coeffs), H, W]
    zernike = zernike.unsqueeze(0)

    # convert to Pytorch Complex tensor
    real, imag = utils.polar_to_rect(torch.ones_like(zernike), zernike)
    return torch.complex(real, imag)


def compute_zernike_basis(num_polynomials, field_res, dtype=torch.float32, wo_piston=False):
    """Computes a set of Zernike basis function with resolution field_res

    num_polynomials: number of Zernike polynomials in this basis
    field_res: [height, width] in px, any list-like object
    dtype: torch dtype for computation at different precision
    """

    # size the zernike basis to avoid circular masking
    zernike_diam = int(np.ceil(np.sqrt(field_res[0]**2 + field_res[1]**2)))

    # create zernike functions

    if not wo_piston:
        zernike = zernikeArray(num_polynomials, zernike_diam)
    else:  # 200427 - exclude pistorn term
        idxs = range(2, 2 + num_polynomials)
        zernike = zernikeArray(idxs, zernike_diam)

    zernike = utils.crop_image(zernike, field_res, pytorch=False)

    # convert to tensor and create phase
    zernike = torch.tensor(zernike, dtype=dtype, requires_grad=False)

    return zernike


def add_noise():
    # torch.poisson
    pass


def _propagate_field_imaging(pupil_phase, pupil_amp, padding_factor=2,
                    photon_flux=None, crop=None):
    pupil_phase = torch.Tensor(pupil_phase)
    pupil_amp = torch.Tensor(pupil_amp)
    Efield = pupil_amp * torch.exp(1j * pupil_phase)
    # pad image to obtain given pixel scale
    target_shape = np.array(np.round(np.array(Efield.shape) * (padding_factor+1)), dtype='int')
    pad = list((target_shape - np.array(Efield.shape)))*2
    # efield_padded = pad_image(Efield, target_shape//2)
    efield_padded = nn.functional.pad(Efield, pad, mode='constant', value=0)

    # Fourier transform
    ufield = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(efield_padded)))

    # crop image
    if crop is None:
        ufield_cropped = ufield
    elif crop is True:
        pp = pad[0]
        ufield_cropped = ufield[pp:-pp, pp:-pp]
    elif crop > 1:
        ufield_cropped = crop_image(ufield, crop, pytorch=False)
        

    # calculate PSF
    psf = torch.abs(ufield_cropped)**2

    psf /= torch.sum(psf)
    if photon_flux is not None:
        # adding photon noise noise
        #psf_noise = psf / torch.sum(psf)
        psf_noise = torch.poisson(photon_flux * psf)

        psf_noise /=torch.sum(psf_noise)
        # return psf
        return psf_noise
    else:
        return psf

def propagate_field(pupil_phase, pupil_amp, padding_factor=2,
                    photon_flux=None, vortex=None, crop=None):
    if vortex is None or vortex is False:
        return _propagate_field_imaging(pupil_phase, pupil_amp, padding_factor=padding_factor,
                    photon_flux=photon_flux, crop=crop)
    else:
        assert vortex._get_name() is 'Vortex'
        psf = vortex.coronagraph(wavefront=pupil_phase,
                           ideal_vortex=vortex.idealVortex,
                           Enum=vortex.Enum,
                           Eperf=vortex.Eperf,
                           apply_rejection=True,
                           rejected_psf=vortex.rejectedPsf)
        psfs_crop = crop_array(array=psf,
                               center=(vortex.gsize_pad / 2, vortex.gsize_pad / 2),
                               bbox=vortex.det_size)
        return psfs_crop



def propagate_field_with_zernike(pupil_amp, zernCoeffs, zernBasis=None,
                                 padding_factor=2,
                                 photon_flux=None, vortex=None):

    if vortex is None or vortex is False:
        if zernBasis is None:
            zernBasis = torch.Tensor(aotools.zernikeArray(1 + len(zernCoeffs),
                                                        pupil_amp.shape[0])[1:])
        if type(pupil_amp) is not torch.Tensor:
            pupil_amp = torch.Tensor(pupil_amp)
        if type(zernCoeffs) is not torch.Tensor:
            zernCoeffs = torch.Tensor(zernCoeffs)
        assert zernBasis.shape[0] == zernCoeffs.shape[0]
        zernCoeffs = torch.reshape(zernCoeffs, (zernCoeffs.shape[0], 1, 1))
        pupil_phase = (zernCoeffs * zernBasis).sum(0, keepdim=True).squeeze(0)

        return propagate_field(pupil_phase, pupil_amp,
                            padding_factor=padding_factor,
                            photon_flux=photon_flux, vortex=None)
    else:
        assert vortex._get_name() is 'Vortex'
        return vortex(coeffs=zernCoeffs, ao_residuals=False)[0,0]


#


def show():
    from astropy.visualization import imshow_norm,\
        SqrtStretch, MinMaxInterval, PercentileInterval, ManualInterval,\
        LinearStretch, SinhStretch, LogStretch

    myStretch = LogStretch()
    plt.figure()
    imshow_norm(psf, plt.gca(), stretch=myStretch)


# MAIN
    # zernBasis = aotools.zernikeArray(1, 128)
    # ap = zernike_basis[0]
