import sys
sys.path.append('../../')

import torch
from torch.fft import ifft2, ifftshift, fft2, fftshift
import torch.nn.functional as F
import functorch

import numpy as np
import matplotlib.pyplot as plt
from psi.instruments import GenericInstrument
from psi.helperFunctions import LazyLogger
from simulator_with_vortex import Vortex, crop_array
import hcipy
import aotools
from astropy.visualization import imshow_norm, SqrtStretch, ManualInterval, MinMaxInterval, PercentileInterval


class ADInstrument(GenericInstrument):
    '''
        Instrument with 'built-in' automatic differentiation.


        HISTORY
        2023-03-17 : built around the Vortex class upgraded by Maxime Q. to be AD
    '''
    def __init__(self, conf, logger=LazyLogger('ADSim')):
        super().__init__(conf, conf.tel_diam)

        self.logger = logger
        self._setup(conf)

    def _setup(self, conf):
        self.wavelength = conf.wavelength
        self.diam = conf.tel_diam
        self._inst_mode = conf.inst_mode
        self._asym_stop = conf.asym_stop
        if self._asym_stop:
            self._asym_angle = conf.asym_angle
            self._asym_width = conf.asym_width
        self._vc_charge= conf.vc_charge
        self._vc_vector = conf.vc_vector
        self.photon_flux = 1e6
        self.ref_psf = 0

        self._padded_pupil_grid_size = int(conf.det_res * conf.npupil)
        self.det_size = 2 * self._focal_grid_size * self._focal_grid_resolution
        self.nmodes = 21

        # for 'STD' pupil
        self.cobs = 0.3 
        self.spider_width = 0.1
        self.flyot=0.95

        if (self._inst_mode != 'CVC') and (self._inst_mode != 'RAVC'):
            self._vc_charge = 0

        # --  Setting up the entrance pupil --

        if conf.pupil == 'CIRC':
            self.aperture = hcipy.aperture.make_circular_aperture(self.diam)(self.pupilGrid)
        elif conf.pupil == 'STD':
            self.aperture = hcipy.aperture.make_obstructed_circular_aperture(self.diam,
                                                             self.cobs,
                                                             num_spiders=4,
                                                             spider_width=self.spider_width)(self.pupilGrid)

        #  -- Setting up the asymmetric stop
        if self._inst_mode == 'ELT' and self._asym_stop:
            self.add_asymmetry_stop(which='pupil')

        elif self._inst_mode == 'CVC':
            if conf.pupil == 'CIRC':
                self.lyot_stop_mask = hcipy.aperture.make_circular_aperture(self.diam * self.flyot)(self.pupilGrid)
            elif conf.pupil == 'STD':
                self.lyot_stop_mask =  hcipy.aperture.make_obstructed_circular_aperture(self.flyot * self.diam,
                                                                self.cobs,
                                                                num_spiders=4,
                                                                spider_width=self.flyot * self.spider_width)(self.pupilGrid)
            self.add_asymmetry_stop(which='lyot')

    def asymmetry_generator(self):
        raise NotImplemented

    def add_asymmetry_stop(self, which='pupil'):
        # spider_gen = hcipy.make_spider_infinite((-2,0),
        #                                             self._asym_angle,
        #                                             self._asym_width * self.diam)
        spider_gen = hcipy.make_spider((-0, 0), (-2,0),
                                        self._asym_width * self.diam)
        asym_arm = spider_gen(self.pupilGrid)

        # spider_gen_2 = hcipy.make_spider_infinite((0,-2),
        #                                         self._asym_angle+90,
        #                                         self._asym_width * self.diam)
        spider_gen_2 = hcipy.make_spider((0, -0), (0,-2),
                                                self._asym_width * self.diam)
        asym_arm_2 = spider_gen_2(self.pupilGrid)
         
        if which is 'pupil':
            self.aperture *= asym_arm
            # self.aperture *= asym_arm_2
        elif which is 'lyot':
            self.lyot_stop_mask *= asym_arm
            self.lyot_stop_mask *= asym_arm_2
            # self.aperture *= asym_arm
            # self.aperture *= asym_arm_2


        # TODO: clean that mess 
        self.aperture_torch = torch.Tensor(np.array(self.aperture.shaped))

    def build_optical_model(self):
        if self._inst_mode == 'CVC':
            assert self._vc_charge == 2 or self._vc_charge == 4
        else:
            assert self._vc_charge == 0

        conf = {
            'nmodes': self.nmodes,  # Number of Zernike coefficients
            'first_poly_index' : 2,
            'lam': self.wavelength,  # observed wavelength (in meter)
            'pupil_size': self._size_pupil_grid,  # entrance pupil size
            'N': self._padded_pupil_grid_size,  # nb of grid points -> padded gridsize
            'L': int(self.diam / (self._size_pupil_grid / self._padded_pupil_grid_size)),  # physical size of grid [m] -> L/D = 4 pix sous tache Airy
            'D': self.diam,  # diameter of the telescope
            'lp': torch.tensor(self._vc_charge),  # Topological charge of the Vortex coronagraph
            'cobs': 0.0, #.3,  # central obstruction
            'fLyot': 1.,  # the Lyot pupil resizing factor
            'ndet': self.det_size,  # Size of the detector in pixels
            'Rej': None,  # .75,
            }

        self.optical_model = Vortex(config=conf,
                            lp=conf["lp"],
                            generator_device="cpu")

        
        self.optical_model.setPupilStop(self.aperture.shaped)
        if self._inst_mode == 'CVC':
            self.optical_model.setLyotStop(self.lyot_stop_mask.shaped)
            if self._vc_vector:
                self.optical_model_2 = Vortex(config=conf,
                                            lp=-conf["lp"],
                                            generator_device="cpu")
                self.optical_model_2.setPupilStop(self.aperture.shaped)
                self.optical_model_2.setLyotStop(self.lyot_stop_mask.shaped)
        else:
            self.optical_model.setLyotStop(self.aperture.shaped)

    def propagate_forward(self, phase):
        psf = self.optical_model.coronagraph(wavefront=phase,
                                            ideal_vortex=self.optical_model.idealVortex,
                                            Enum=self.optical_model.Enum,
                                            Eperf=self.optical_model.Eperf,
                                            apply_rejection=True,
                                            rejected_psf=self.optical_model.rejectedPsf)
        psfs_crop = crop_array(array=psf,
                               center=(self.optical_model.gsize_pad / 2, self.optical_model.gsize_pad / 2),
                               bbox=self.optical_model.det_size)
        return psfs_crop

    # def propagate_to_fourier_plane(self, phase):
    #     psf = self.optical_model.coronagraph(wavefront=phase,
    #                                         ideal_vortex=self.optical_model.idealVortex,
    #                                         Enum=self.optical_model.Enum,
    #                                         Eperf=self.optical_model.Eperf,
    #                                         apply_rejection=True,
    #                                         rejected_psf=self.optical_model.rejectedPsf)
    #     efield = fftshift(fft2(psf))
    #     return efield
    def calculate_ref_psf(self):
        self.ref_psf = self.propagate_forward_with_zernike(torch.zeros(self.nmodes), False)

    # def add_noise(self, psf):
    #     psf = psf / torch.sum(psf)
    #     psf = torch.poisson(self.photon_flux * psf)
    #     psf = psf/torch.sum(psf)
    #     return psf

    def propagate_forward_with_zernike(self, modal_coefficients, photon_noise):
        if len(modal_coefficients) != self.nmodes:
            self.logger.warn('Number of requested modal coefficients is not the same as in the optical model.')
        modal_coefficients = torch.Tensor(modal_coefficients)
        psf = self.optical_model(coeffs=modal_coefficients, ao_residuals=False, crop=True)[0,0]
        if self._inst_mode == 'CVC' and self._vc_vector:
            psf_2 = self.optical_model_2(coeffs=modal_coefficients, ao_residuals=False, crop=True)[0,0]
            psf = (psf + psf_2) / 2
        if photon_noise:
            psf = psf / torch.sum(psf)
            psf = torch.poisson(self.photon_flux * psf)
        psf = psf/torch.sum(psf)
        #print(psf.shape)
        return psf - self.ref_psf
    
    def propagate_to_fourier_plane_with_zernike(self, modal_coefficients, photon_noise):
        if len(modal_coefficients) != self.nmodes:
            self.logger.warn('Number of requested modal coefficients is not the same as in the optical model.')
        modal_coefficients = torch.Tensor(modal_coefficients)
    
        psf = self.optical_model(coeffs=modal_coefficients, ao_residuals=False, crop=False)[0]
        if photon_noise:
            psf = psf / torch.sum(psf)
            psf = torch.poisson(self.photon_flux * psf)
        psf = psf/torch.sum(psf)

        pp = psf.shape[0] // 2
        psf_padded = F.pad(psf, (pp, pp, pp, pp))
        #print(psf.shape)
        #print(psf_padded.shape)
        efield = ifftshift(ifft2(fftshift(psf_padded) * np.exp(1j * 0)))

        rr = self._size_pupil_grid #self._size_pupil_grid * 2 - 2 
        cc = aotools.circle(rr, efield.shape[0])
        self.fourier_efield = efield
        self.fourier_efield[cc!=1] = torch.nan

        return torch.angle(self.fourier_efield)
    
    #-------------------------------------------#
    def compute_jacobian(self, which='modal', plane='fourier'):
        '''
            TODO should be part of a AD Sensor and not of an instrument
        '''
        if which=='modal':
            zern_coeffs = torch.zeros(self.nmodes) #coeffs[0] * 0
            if plane == 'fourier':
                self.jacobian = functorch.jacfwd(self.propagate_to_fourier_plane_with_zernike,
                                                 randomness='different')(zern_coeffs, False)
                self.jacobian_matrix = self.jacobian[torch.isnan(self.jacobian)==False]
                ll = self.jacobian_matrix.shape[0] // self.nmodes
                self.jacobian_matrix = self.jacobian_matrix.reshape((ll, self.nmodes))
            elif plane == 'img':
                self.jacobian = functorch.jacfwd(self.propagate_forward_with_zernike,
                                                 randomness='different')(zern_coeffs, False)
                self.jacobian_matrix = self.jacobian.reshape((self.det_size * self.det_size, self.nmodes)) 

            self.jacobian_matrix_inverse = torch.pinverse(self.jacobian_matrix)

    # def zero_wavefront(self, plane='fourier'):
    #     coeffs = torch.zeros(inst.nmodes)
        

    def test_wavefront_retrieval(self, modal_coefficients, plane='fourier', display=True):
        zernBasis = self.optical_model.zernike_basis
        aberration = np.dot(modal_coefficients[:].T, zernBasis.reshape((self.nmodes, self._size_pupil_grid**2)))\
            .reshape((self._size_pupil_grid, self._size_pupil_grid))
        
        if plane is 'fourier':
            meas_2d = self.propagate_to_fourier_plane_with_zernike(modal_coefficients, True)
            meas = meas_2d[torch.isnan(meas_2d)==False]
        elif plane is 'img':
            meas_2d = self.propagate_forward_with_zernike(modal_coefficients, True) 
            meas = meas_2d.flatten()

        c_retrieved = torch.matmul(self.jacobian_matrix_inverse, meas).detach().numpy()
        wf_retrieved = np.dot(c_retrieved[:].T, zernBasis.reshape((self.nmodes, self._size_pupil_grid**2)))
        wf_retrieved = np.reshape(wf_retrieved, ((self._size_pupil_grid, self._size_pupil_grid))) 

        if self._inst_mode == 'CVC':
            aperture = self.aperture.shaped * self.lyot_stop_mask.shaped
        else:
            aperture = self.aperture.shaped
        if display:
            plt.figure()
            plt.subplot(131)
            plt.imshow(aperture * aberration)
            plt.subplot(132)
            # plt.imshow(meas_2d)
            imshow_norm(meas_2d.detach().numpy(), interval=PercentileInterval(99))
            plt.subplot(133)
            plt.imshow(aperture* wf_retrieved)

            plt.figure()
            plt.plot(modal_coefficients, label='injected')
            plt.plot(c_retrieved, label='retrieved')
            plt.legend()
        # return wf_retrieved

if __name__ == '__main__':
    '''
    TODO test different modal basis (e.g. Gendrinou)
    '''
    from psi.configParser import loadConfiguration

    
    config_file = 'config_ADsim.py'
    cfg = loadConfiguration(config_file)
    inst = ADInstrument(cfg.params)
    inst.build_optical_model()

    coeffs = torch.zeros(inst.nmodes)
    # coeffs[0:6] = torch.Tensor([0., 0., 0., 0., 0., 0.1])
    coeffs[0:10] = torch.Tensor([0., 0., -0., 0.3, 0.0, 0.,0.,0,0,0.])
    # # psf = inst.propagate_forward_with_zernike(coeffs)
    # # plt.figure()
    # # plt.imshow(psf)
    # fourier_phase = inst.propagate_to_fourier_plane_with_zernike(coeffs, True)
    # # rr = efield.shape[0] // 4 - 2
    # # cc = aotools.circle(rr, efield.shape[0])
    # # efield[cc!=1] = np.nan

    # fourier_phase_n = inst.propagate_to_fourier_plane_with_zernike(-coeffs, True)
    # # efield_n[cc!=1] = np.nan

    # plt.figure()
    # plt.subplot(121)
    # imshow_norm(fourier_phase, interval=PercentileInterval(95))
    # plt.subplot(122)
    # imshow_norm(fourier_phase_n, interval=PercentileInterval(95))
    inst.photon_flux = 1e6
    inst.calculate_ref_psf()
    inst.compute_jacobian(plane='img')
    inst.test_wavefront_retrieval(coeffs, plane='img')

    # cov = np.dot(inst.jacobian_matrix_inverse, inst.jacobian_matrix)
    # plt.figure()
    # plt.imshow(cov, origin='upper')