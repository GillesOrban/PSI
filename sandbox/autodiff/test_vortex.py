from utils import propagate_field, propagate_field_with_zernike
from simulator_with_vortex import Vortex, crop_array
import torch

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import imshow_norm,\
    SqrtStretch, MinMaxInterval, PercentileInterval, ManualInterval,\
    LinearStretch, SinhStretch, LogStretch
import torch.nn.functional as F
import torch.nn as nn
import aotools


verbose = True
direct = True
# photon_flux = 1e6
# nGrid = 128
# nZern = 20
# pad = 1.7689 #4


def make_wavefront(coeffs, vortex):
    W = torch.sum(coeffs[..., None, None] / (2 * np.pi) * vortex.zernike_basis,
                    axis=coeffs.ndim - 1)  # from radian to waves rms

    W_padded = F.pad(W, (vortex.pad, vortex.pad, vortex.pad, vortex.pad, 0, 0),
                        "constant", 0)

    wavefront = W_padded.requires_grad_(True)
    wavefront.retain_grad()
    return wavefront

# -- initialize physical model -- 
diameter = 8
pupil_size = 16 # 64
gridsize = 64  # 256 # padded
wavelength = 2.2 * 1e-6
conf = {
    'nmodes': 10,  # Number of Zernike coefficients
    'lam': wavelength,  # observed wavelength (in meter)
    'pupil_size': pupil_size,  # entrance pupil size
    'N': gridsize,  # nb of grid points -> padded gridsize
    'L': int(diameter / (pupil_size / gridsize)),  # physical size of grid [m] -> L/D = 4 pix sous tache Airy
    'D': diameter,  # diameter of the telescope
    'lp': torch.tensor(2),  # Topological charge of the Vortex coronagraph
    'cobs': 0.3, #.3,  # central obstruction
    'fLyot': .98,  # the Lyot pupil resizing factor
    'ndet': 64,  # Size of the detector in pixels
    'Rej': None,  # .75,
}

# coeffs = torch.tensor([[0., -.3, 0., 0., 0., 0, 0, 0, 0, 0],
#                         [.1, .05, 0., 0., .1, 0, 0, 0, 0, 0],
#                         [0.4, 0.3, 0.5, .1, .5, .4, .1, .05, .04, .09]])  # in radians
coeffs = torch.tensor([[0., 0.75, 0., 0.25, 0., 0, 0, 0, 0, 0]])  # in radians

conf["first_poly_index"] = 4  # defocus

vortex = Vortex(config=conf,
                    lp=conf["lp"],
                    generator_device="cpu")
vortex_d = Vortex(config=conf,
            lp=-conf["lp"],
            generator_device="cpu")


# -- Create aberrated PSFs with lp +2 and lp -2
phase = make_wavefront(coeffs, vortex)
psf_p = propagate_field(phase, vortex.pupilStop,
                    photon_flux=None, vortex=vortex)
psf_m = propagate_field(phase, vortex.pupilStop,
                    photon_flux=None, vortex=vortex_d)

plt.figure()
plt.subplot(121)
plt.imshow(psf_p[0,0,:].detach().numpy())
plt.subplot(122)
plt.imshow(psf_m[0,0,:].detach().numpy())

# Stochastic gradient descent
# num_iters = 61
# phase_sgd = SGD(vortex.pupilStop, 0,
#                 vortex.pad, num_iters, useZernCoeffs=True,
#                 loss=nn.MSELoss(), lr=0.2, lr_s=0, s0=1.0,
#                 device=torch.device('cpu'))

# final_phase = phase_sgd(psf_p, psf_m,
#                         init_phase=phase * 0, photon_flux=None)
# check_psf = propagate_field(final_phase.detach().numpy(), vortex.pupilStop,
#                                     photon_flux=None)

zernBasis = aotools.zernikeArray(coeffs.nelement() + conf['first_poly_index'] + 1, pupil_size)
aperture = torch.Tensor(zernBasis[0]) * 1
zernBasis = torch.Tensor(zernBasis[1:])
# create asymmetric aperture
p = pupil_size //2 -1
aperture[p:p+2,:p] = 0
aperture[:p-p//2, p:p+2] = 0

def func(phase):
    '''
        to define phase and aperture, see 'stochastic_graident_descent.py' that uses aotools
        TODO kwargs in func does not seem to work...
    '''
    return propagate_field(phase, aperture, photon_flux=None, vortex=None, crop=32)

def func_zernike(zern_coeffs):
    return propagate_field_with_zernike(aperture, zern_coeffs, zernBasis=zernBasis[:len(zern_coeffs)],
                                         photon_flux=None, vortex=vortex)

def func_vortex(phase):
    phase_padded = F.pad(phase[np.newaxis,:], (vortex.pad, vortex.pad, vortex.pad, vortex.pad, 0, 0),
                    "constant", 0)
    return propagate_field(phase_padded, vortex.pupilStop, photon_flux=None, vortex=vortex)[0]

# def func_vortex_zernike(zern_coeffs):
#     return propagate_field_with_zernike(aperture, zern_coeffs, zernBasis=zernBasis[:len(zern_coeffs)],
#                                          photon_flux=None, vortex=None)

import functorch

# --- model --
aberration = zernBasis[3] * 0
test_psf = func(aberration)
fp_size = test_psf.shape[0]

# 'zonal'
# jacobian = functorch.jacfwd(func)(aberration)
# jacobian_matrix = jacobian.reshape((fp_size**2, pupil_size**2))

# 'modal'
nmodes=10
zern_coeffs = torch.zeros(nmodes) #coeffs[0] * 0
jacobian = functorch.jacfwd(func_zernike)(zern_coeffs)
test_psf = func_zernike(zern_coeffs)
xdim = test_psf.shape[0]
ydim = test_psf.shape[1]
jacobian_matrix = jacobian.reshape((xdim*ydim, nmodes))


# --- test --
# aberration = zernBasis[6]*2 #+ zernBasis[7] *0.1
# psf = func(aberration)
# wf_retrieved = np.dot(jacobian_matrix.T, psf.flatten()).reshape((pupil_size, pupil_size))

# modal case
coeffs = torch.zeros(nmodes)
# coeffs[2:6] = torch.Tensor([0.1, 0.05, 0.05, 0.05])
coeffs[3]=0.1
aberration = np.dot(coeffs[:].T, \
                    zernBasis[:nmodes].reshape((nmodes, pupil_size**2)))\
            .reshape((pupil_size, pupil_size))

psf = func_zernike(coeffs)

c_retrieved = np.dot(jacobian_matrix.T, psf.flatten())
wf_retrieved = np.dot(c_retrieved[:].T, zernBasis[:nmodes].reshape((nmodes, pupil_size**2)))
wf_retrieved = np.reshape(wf_retrieved, ((pupil_size, pupil_size)))

plt.figure()
plt.subplot(131)
plt.imshow(aperture * aberration)
plt.subplot(132)
plt.imshow(psf)
plt.subplot(133)
plt.imshow(aperture * wf_retrieved)