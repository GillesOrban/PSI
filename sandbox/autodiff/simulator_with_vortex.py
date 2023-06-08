import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import aotools
import poppy
import numpy as np
import matplotlib.pyplot as plt


class Vortex(nn.Module):
    def __init__(self, config, lp, generator_device, shift=0):
        '''
            Simulate the vector vortex coronagraph.

            Doesn't include the effect of the opaque central spot of the VVC,
            neither the chromatic effect (leakage term).
            See Krist, ..., Mawet, 2013 "Assessing the performance
            limits of internal coronagraph through end-to-end modeling"

            parameters
            -----------
            config: dictionary with the simulation parameter"
                'lam': wavelength [m]
                'pupil_size': pixel size of the entrance pupil
                'cobs': size fraction taken by the central obstruction (between 0 and 1)
                'ndet': pixel size of the detector
                'nmodes': number of Zernike modes
                'first_poly_index': first polynomial index following Noll convention (e.g., 4 if starting from defocus)
                'N' : nb of grid points
                'L' : physical size of grid [m]
                'D' : telescope diameter [m]
                'lp': the vortex charge (2, 4, etc.)
                'fLyot': the Lyot pupil resizing factor
                'Rej' : (optional) simple way to simulate the leakage term of the coronagraph
                       if not None:
                        the E field in the coronagraph plane is :
                         Ecorr += 1/ sqrt(rejection) * Efield....
            fLyot: Lyot stop as fraction of diameter
            cobs: central obscuration as fraction of diameter

            Simulation parameter -- sampling
            =================================
                L/D define the padding and thus the sampling of the PSF:
                    L / D = lbda / D   (in pixels)
                so diameter of the first Airy null: 2.44 L / D

            *  Alternative pupil and image grid spacing definition *
            For a given pixel scale [''/px], one wants:
                imgGridSpacing = (pxs/206265) * lbda    [m]
                pupilGridSpacing= 1 / (N * imgGridSpacing)    [m^-1]
        '''
        super(Vortex, self).__init__()

        self.wavelength = config["lam"]
        self.pupil_size = config["pupil_size"]
        self.det_size = config["ndet"]
        self.nb_modes = config["nmodes"]
        first_ind = config["first_poly_index"] - 1
        cobs = config["cobs"]
        N = config['N']     # Number of grid points
        L = config['L']     # physical size of grid [m]
        self.D = config['D']  # tel. diameter in [m]
        self.pupilGridSpacing = L / N   # physical size of pixel [m]
        self.imgGridSpacing = 1 / (N * self.pupilGridSpacing)
        self.flyot = config['fLyot']
        self.lp = lp

        self.gsize_pad = N
        self.pad = (config['N'] - self.pupil_size) // 2
        self._generator_device = generator_device

        ff = torch.arange(-N / 2. + shift, N / 2. + shift) * self.pupilGridSpacing
        x_tp, y_tp = torch.meshgrid(ff, ff)
        x, y = (x_tp.T, y_tp.T)  # to keep matrix order given by np.meshgrid()

        self.r, self.theta = cart2pol(x, y)

        idealVortex = torch.exp(torch.tensor(1.j) * self.lp * self.theta)
        idealVortex[self.r == 0] = 0

        if 'Rej' in config.keys():
            self.rejection = config['Rej']
        else:
            self.rejection = None
        self.rejectedPsf = 0

        self.circStop = torch.from_numpy(aotools.circle(self.pupil_size, self.gsize_pad))

        self.centralObscuration = cobs
        circ_rad = self.pupil_size // 2
        pupilStop = aotools.circle(circ_rad, self.gsize_pad)
        lyotStop = aotools.circle(circ_rad * self.flyot, self.gsize_pad)
        if config['fLyot'] is not None and cobs is not None:
            # add central obstruction:
            pupilStop -= aotools.circle(circ_rad * cobs, self.gsize_pad)
            lyotStop -= aotools.circle(circ_rad * cobs / self.flyot, self.gsize_pad)
        else:
            print("WARNING: wrong pupil / lyot definition")

        if cobs != 0.:
            pupil_np = np.array(crop_array(array=torch.from_numpy(pupilStop),
                                           center=(self.gsize_pad / 2, self.gsize_pad / 2),
                                           bbox=self.pupil_size)).squeeze()
            # Define the Zernike basis:
            zern_basis_np = poppy.zernike.arbitrary_basis(aperture=pupil_np,
                                                          nterms=self.nb_modes + first_ind,
                                                          outside=0.
                                                          )[first_ind:, :, :]
        else:
            # Define the Zernike basis:
            zern_basis_np = aotools.zernikeArray(J=self.nb_modes + first_ind,
                                                 N=int(N * self.D / L),
                                                 norm='rms'
                                                 )[first_ind:, :, :]

        # Define perfect coronagraph:
        Enum, Eperf = self.perfect(idealVortex)

        # Compute for rejected PSF:
        if self.rejection is not None:
            idealVortex_lp0 = torch.exp(torch.tensor(1.j) * 0. * self.theta)
            idealVortex_lp0[self.r == 0] = 0
            Enum_lp0, Eperf_lp0 = self.perfect(idealVortex_lp0)

            self.idealVortex_lp0 = idealVortex_lp0.to(self._generator_device)
            self.Enum_lp0 = Enum_lp0.to(self._generator_device)
            self.Eperf_lp0 = Eperf_lp0.to(self._generator_device)

        # if torch.cuda.device_count() > 1:
        #     # self.register_buffer() -> for training with multiple GPUs
        #     self.register_buffer('zernike_basis', torch.from_numpy(zern_basis_np).float())
        #     self.register_buffer('pupilStop', pupilStop)
        #     self.register_buffer('lyotStop', lyotStop)
        #     self.register_buffer('idealVortex', idealVortex)
        #     self.register_buffer('Enum', Enum)
        #     self.register_buffer('Eperf', Eperf)
        # else:
        self.zernike_basis = torch.from_numpy(zern_basis_np).float().to(self._generator_device)
        self.pupilStop = torch.from_numpy(pupilStop).to(self._generator_device)
        self.lyotStop = torch.from_numpy(lyotStop).to(self._generator_device)
        self.idealVortex = idealVortex.to(self._generator_device)
        self.Enum = Enum.to(self._generator_device)
        self.Eperf = Eperf.to(self._generator_device)

    def setModalBasis(self, modal_basis):
        raise NotImplementedError

    def setPupilStop(self, pupilStop):
        '''
            pupilStop should be unpadded
        '''
        assert pupilStop.shape == self.zernike_basis.shape[1:]
        if isinstance(pupilStop, np.ndarray):
            pupilStop = torch.from_numpy(pupilStop).to(self._generator_device)
        self.pupilStop = F.pad(pupilStop, (self.pad, self.pad, self.pad, self.pad),
                         "constant", 0)
        # self.pupilStop = torch.from_numpy(aperture).to(self._generator_device)
   
    def setLyotStop(self, lyotStop):
        '''
            lyotStop should be unpadded
        '''
        assert lyotStop.shape == self.zernike_basis.shape[1:]
        if isinstance(lyotStop, np.ndarray):
            lyotStop = torch.from_numpy(lyotStop).to(self._generator_device)
        self.lyotStop = F.pad(lyotStop, (self.pad, self.pad, self.pad, self.pad),
                         "constant", 0)
        # self.lyotStop = torch.from_numpy(aperture).to(self._generator_device)

    def getAperture(self):
        aperture = crop_array(array=self.pupilStop,
                              center=(self.gsize_pad / 2, self.gsize_pad / 2),
                              bbox=self.pupil_size).squeeze(0)
        return aperture
    
    def getPupilStop(self):
        return self.getAperture()
    
    def getLyotStop(self):
        aperture = crop_array(array=self.lyotStop,
                              center=(self.gsize_pad / 2, self.gsize_pad / 2),
                              bbox=self.pupil_size).squeeze(0)
        return aperture

    def _getSimulatedPixelScale(self, wvl):
        return self.imgGridSpacing * 206265 * wvl

    def perfect(self, idealVortex):
        '''
            Define 2 variables:
            Enum    : E field in focal plane before coronagraph of a flat wave
            Eperf   : E field in focal plane after coronagraph
                to obtain perfect suppression with a circular pupil.

            Eperf is valid for a perfect vortex and circular pupil
        '''
        Enum = ft2(self.circStop * torch.exp(torch.tensor(1.j) * 0),
                   self.pupilGridSpacing)
        Eperf = ft2(ift2(Enum * idealVortex,
                         self.imgGridSpacing) * (1 - self.circStop),
                    self.pupilGridSpacing)
        return Enum, Eperf

    def coronagraph(self, wavefront, ideal_vortex, Enum, Eperf, apply_rejection, rejected_psf=0):
        '''
            Calculate the variables
            -------------------------
            Efocal : E field aberrated in focal plane before vortex
            Ecorr  : E field in focal plane after coronagraph corrected
                for numerical inaccuracy
            Elyot  : E field in Lyot plane before lyot stop
            Efinal : E field in image plane
            PSF    : |Efinal|^2

            parameter:
            wavefront: aberration (pupil plane)

            vortex : focal plane mask. By default this is self.getVortex()

            Returns
            -------
            PSF  : |Efinal|^2
        '''

        self.Efocal = ft2(self.pupilStop * torch.exp(torch.tensor(1.j) * 2 * math.pi * wavefront),
                          self.pupilGridSpacing)

        if (torch.remainder(self.lp, 2) == 0) and (self.lp != 0):   # torch.remainder() == np.mod()
            Ecorr = (self.Efocal - Enum) * ideal_vortex + Eperf
        else:
            Ecorr = self.Efocal * ideal_vortex

        self.Elyot = ift2(Ecorr, self.imgGridSpacing)
        self.Efinal = ft2(self.Elyot * self.lyotStop, self.pupilGridSpacing)

        self.psf = torch.abs(self.Efinal) ** 2

        if apply_rejection is True and self.rejection is not None:
            self.psf = self.psf + (rejected_psf / self.rejection)

        return self.psf

    def forward(self, coeffs, ao_residuals, crop=True):
        '''
        '''
        if coeffs.ndim == 1:
            coeffs = coeffs[np.newaxis,:]
        W = torch.sum(coeffs[..., None, None] / (2 * math.pi) * self.zernike_basis, axis=coeffs.ndim - 1)  # from radian to waves rms
        if ao_residuals is not False:
            W += ao_residuals / self.wavelength  # from m to wave

        W_padded = F.pad(W, (self.pad, self.pad, self.pad, self.pad, 0, 0),
                         "constant", 0)

        if self.rejection is not None:
            self.rejectedPsf = self.coronagraph(wavefront=W_padded,
                                                ideal_vortex=self.idealVortex_lp0,
                                                Enum=self.Enum_lp0,
                                                Eperf=self.Eperf_lp0,
                                                apply_rejection=False)

        psfs = self.coronagraph(wavefront=W_padded,
                                ideal_vortex=self.idealVortex,
                                Enum=self.Enum,
                                Eperf=self.Eperf,
                                apply_rejection=True,
                                rejected_psf=self.rejectedPsf)

        if crop:
            psfs = crop_array(array=psfs,
                                center=(self.gsize_pad / 2, self.gsize_pad / 2),
                                bbox=self.det_size)

        return psfs


def ft2(data, delta):
    """
    A properly scaled 2-D FFT

    Parameters:
        data (ndarray): An array on which to perform the FFT
        delta (float): Spacing between elements

    Returns:
        ndarray: scaled FFT
    """

    DATA = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.fftshift(data, dim=(-1, -2))
        ), dim=(-1, -2)
    ) * delta**2

    return DATA


def ift2(DATA, delta_f):
    """
    Scaled inverse 2-D FFT

    Parameters:
        DATA (ndarray): Data in Fourier Space to transform
        delta_f (ndarray): Frequency spacing of grid

    Returns:
        ndarray: Scaled data in real space
    """
    N = DATA.shape[0]
    g = torch.fft.ifftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(DATA, dim=(-2, -1))
        ), dim=(-2, -1)
    ) * (N * delta_f)**2

    return g


def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return(rho, phi)


def crop_array(array, center, bbox, bboxy=None):
    if array.ndim == 2:
        array = array.unsqueeze(0).unsqueeze(0)
    elif array.ndim == 3:
        array = array.unsqueeze(0)

    if bboxy is None:
        bboxy = bbox
    xmin = int(center[0] - bbox / 2)
    xmax = int(center[0] + bbox / 2)
    ymin = int(center[1] - bboxy / 2)
    ymax = int(center[1] + bboxy / 2)
    if xmin < 0:
        xmin = 0
    if xmax > array.shape[-2]:
        xmax = array.shape[-2]
    if ymin < 1:
        ymin = 1
    if ymax > array.shape[-1]:
        ymax = array.shape[-1]
    if ((xmin == 0) | (ymin == 0) | (xmax == array.shape[-2]) | (ymax == array.shape[-1])):
        print(("Warning setting dimension to (xmin,xmax,ymin,ymax)=""({0},{1},{2},{3})".format(xmin, xmax, ymin, ymax)))
    narr = array[:, :, xmin:xmax, ymin:ymax]
    return narr


if __name__ == '__main__':
    diameter = 8
    pupil_size = 64
    gridsize = 256  # padded
    wavelength = 2.2 * 1e-6
    conf = {
        'nmodes': 10,  # Number of Zernike coefficients
        'lam': wavelength,  # observed wavelength (in meter)
        'pupil_size': pupil_size,  # entrance pupil size
        'N': gridsize,  # nb of grid points -> padded gridsize
        'L': int(diameter / (pupil_size / gridsize)),  # physical size of grid [m] -> L/D = 4 pix sous tache Airy
        'D': diameter,  # diameter of the telescope
        'lp': torch.tensor(2),  # Topological charge of the Vortex coronagraph
        'cobs': .3,  # central obstruction
        'fLyot': .98,  # the Lyot pupil resizing factor
        'ndet': 64,  # Size of the detector in pixels
        'Rej': None,  # .75,
    }

    coeffs = torch.tensor([[0., -.3, 0., 0., 0., 0, 0, 0, 0, 0],
                           [.1, .05, 0., 0., .1, 0, 0, 0, 0, 0],
                           [0.4, 0.3, 0.5, .1, .5, .4, .1, .05, .04, .09]])  # in radians

    conf["first_poly_index"] = 4  # defocus

    # Create vortex object to access the pixel scale and aperture:
    # +lp:
    vortex_1 = Vortex(config=conf,
                      lp=conf["lp"],
                      generator_device="cpu")
    # -lp:
    vortex_2 = Vortex(config=conf,
                      lp=-conf["lp"],
                      generator_device="cpu")
    conf["pixel_scale"] = vortex_1._getSimulatedPixelScale(wavelength) * 1e3
    aperture = vortex_1.getAperture().squeeze(0)

    psf_1 = vortex_1(coeffs=coeffs, ao_residuals=False)  # shape: (ao_batch, ncpa_batch, pix, pix)
    psf_2 = vortex_2(coeffs=coeffs, ao_residuals=False)  # shape: (ao_batch, ncpa_batch, pix, pix)

    ind = 0
    data = [aperture, psf_1.squeeze()[ind], psf_2.squeeze()[ind]]
    fig, ax = plt.subplots(1, len(data), figsize=(12, 4))
    for i in range(len(data)):
        im = ax[i].imshow(data[i], origin="lower", cmap="jet")
    plt.show()
