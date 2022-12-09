'''
TODO convert to math form

NOTES ON XARA

Linearized phase relation:
    phase_uv = R^-1 A phase_pup

TFM : TransFer phase matrix = R^-1 A
    TFM = np.diag(1./RED).dot(BLM)
BLM : BaseLine mapping matrix = A




'''


import hcipy
import numpy as np
import matplotlib.pyplot as plt
import sys

import time
import os
from datetime import datetime
import matplotlib.gridspec as gridspec
from astropy.io import fits
import getpass
import datetime
import shutil

import importlib

# sys.path.append('/Users/orban/Projects/METIS/4.PSI/psi_github/')
import psi.psi_utils as psi_utils
from .configParser import loadConfiguration
from .instruments import  CompassSimInstrument, DemoCompassSimInstrument, HcipySimInstrument
from .helperFunctions import LazyLogger, timeit, build_directory_name, copy_cfgFileToDir

from astropy.visualization import imshow_norm,\
    SqrtStretch, MinMaxInterval, PercentileInterval, \
    LinearStretch, SinhStretch, LogStretch


# from config.config_metis_compass import conf

# sys.path.append('/Users/orban/Projects/METIS/4.PSI/legacy_TestArea/')
print(hcipy.__file__)
print(hcipy.__version__)




sys.path.append('/Users/orban/github/')

from xara import xara
from xaosim import xaosim as xs
from scipy.interpolate import griddata


class KernelSensor():
    '''
     Kernel wavefront sensor
     (Asymmetric Fourier Plane WFS)

     Parameters
     ----------
     config_file : str
         filename of the Python config file
     logger : object
         logger object. Default is ``LazyLogger``
    '''
    def __init__(self, config_file, logger=LazyLogger('KERN')):
        self.logger=logger
        self.logger.info('Loading and checking configuration')
        self._config_file = config_file
        self.cfg = loadConfiguration(config_file)

    # def __init__(self):
        # self._pscale = 16.7    # [mas/px]
        # # self._cwavel = 1.6e-6  # central wavelength
        # self._Dtel = 7.92      # telescope diameter

    def setup(self):
        '''
            Setup the PSI wavefront sensor based on the configuration
        '''
        # Build instrument object 'inst'
        self.logger.info('Initialize the instrument object & building the optical model')
        # self.inst = getattr(instruments,
        #                     self.cfg.params.instrument)(self.cfg.params)
        self.inst = eval(self.cfg.params.instrument)(self.cfg.params)
        # importlib.import_module
        self.inst.build_optical_model()
        self._Dtel = self.cfg.params.asym_telDiam
        self._pscale = (self.cfg.params.wavelength / self._Dtel) * 206264.8 / self.cfg.params.det_res * 1e3 # [mas / px]

        #pscale = 4.91168055, #5.32359, #5.47,                      # pixel scale in mas/pix

        # Build or load the Kernel model
        if self.cfg.params.asym_model_fname is None or '':
            self.logger.info('Builing KPI model')
            self._buildModelKPI(fname='toto.fits.gz')
        else:
            self.logger.info('Loading KPI model : {0}'.format(self.cfg.params.asym_model_fname))
            self._loadKPO(fname_model = self.cfg.params.asym_model_fname)


        # Init logging buffer
        self.iter = 0 # iteration index of PSI
        self._loop_stats = []


    def _buildModelKPI(self, fname="scexao_asym.fits.gz", verbose=False):
        '''
            TODO KPI model with finer pupil image ? -> impact on accuracy and matrix sizes ?
            TODO Adapt bmax in very low SNR cases ? -- check 1.1*Dtel is optimum (or better 1)
        '''
        # Create model
        pscale = self._Dtel / self.inst._size_pupil_grid
        step= self._Dtel / self.cfg.params.asym_nsteps # [meter] # 0.3
        tmin = self.cfg.params.asym_tmin # 0.1 # cut-off value for transmissive or not
        model  = xara.core.create_discrete_model(self.inst.aperture.shaped,
                                                 pscale,
                                                 step,
                                                 binary=False,
                                                 tmin=tmin)
        # bmax : safeguard that tells the KPO data structure creation function to
        #         discard any resulting baseline that would be greater than the provided value
        kpi = xara.KPI(array=model, bmax=self._Dtel*1.1) # TBC -> bmax needed ??

        # Save model
        self.logger.info('Saving KPI model to {0}'.format(fname))
        kpi.save_to_file(fname)

        # Display
        if verbose:
            kpi.plot_pupil_and_uv()
    
    def _loadKPO(self, fname_model="scexao_asym.fits.gz"):
        '''
            fname_model : .fits or .fits.gz

            TFM : Transfer phase Matrix = R^-1 A
        '''
        self.kpo    = xara.KPO(fname=fname_model) # load the KPO structure
        U, S, Vt = np.linalg.svd(self.kpo.kpi.TFM, full_matrices=0)

        # filter some of the low singular values
        neig = int(0.4*len(S))     # 200 nomber of singular values to keep (max=509 for this model)
        self.logger.info('Keeping 40% of the eigenvalues : {0}'.format(neig))
        Sinv = 1 / S
        Sinv[neig:] = 0.0

        # computation of the pseudo inverse
        self.phase_tf_inv = Vt.T.dot(np.diag(Sinv)).dot(U.T)

        # define small aperture 
        nbs = self.cfg.params.asym_nsteps
        asym_telDiam = self.cfg.params.asym_telDiam
        step_size = asym_telDiam / nbs

        self._small_aperture = np.zeros((nbs-1, nbs-1))
        vac_coords = self.kpo.kpi.VAC
        coords = np.array(vac_coords[:,0:2] / step_size + \
                    (asym_telDiam/2) / step_size, dtype='int')
        self._small_aperture[list(coords[:,1]-1), list(coords[:,0]-1)] = vac_coords[:,2]

    def computeWavefront(self, img):
        # TODO empty kpo.CVIS (and other arrays) that use unnecessarily memory

        self.kpo.extract_KPD_single_frame(img, self._pscale,
                                          self.inst.wavelength, recenter=False)
        self._wft  = -self.phase_tf_inv.dot(np.angle(self.kpo.CVIS[-1][0]))     # compute a wavefront

        # ISZ = img.shape[0]
        # cwavel = self.inst.wavelength
        # m2pix  = xara.core.mas2rad(self._pscale) * ISZ / cwavel
        # self.logger.info('m2pix = {0}'.format(m2pix))
        #
        # cvis = self.kpo.extract_cvis_from_img(img, m2pix)
        # self._wft  = -self.phase_tf_inv.dot(np.angle(cvis))     # compute a wavefront


        # Project on 2D map
        # self._interpolateWavefrontOn2D(self.kpo.kpi.VAC, self._wft)
        self._convertTo2D(self.kpo.kpi.VAC, self._wft)

    def _convertTo2D(self, vac_coords, wf_1d):
        '''
            TODO if asym_nsteps is odd, phase should be dim nbs x nbs
        '''
        asym_telDiam = self.cfg.params.asym_telDiam
        asym_nsteps = self.cfg.params.asym_nsteps

        step_size = asym_telDiam / asym_nsteps

        nbs=asym_nsteps
        phase = np.zeros((nbs-1, nbs-1))

        coords = np.array(vac_coords[:,0:2] / step_size + \
                         (asym_telDiam/2) / step_size, dtype='int')
        phase[list(coords[:,1]-1), list(coords[:,0]-1)] = np.append(0, self._wft)#vac_coords[:,2]

        # self._small_aperture = np.zeros((nbs-1, nbs-1))
        # self._small_aperture[list(coords[:,1]-1), list(coords[:,0]-1)] = vac_coords[:,2]

        self.wavefront = phase


    # def _interpolateWavefrontOn2D(self, vac_coords, wf_1d):
    #     nbs = np.int(np.sqrt(len(wf_1d)/np.pi) * 2 + 1 )
    #
    #     Nx = nbs
    #     Ny = nbs
    #     ncoordX = (np.arange(Nx) - Nx / 2. + 0.5) / (Nx / 2.) * self._Dtel / 2
    #     ncoordY = (np.arange(Ny) - Ny / 2. + 0.5) / (Ny / 2.) * self._Dtel / 2
    #
    #     # x, y = np.meshgrid(coordX, coordY)
    #     xnew, ynew = np.meshgrid(ncoordX, ncoordY)
    #
    #     # self._virtual_aperture = griddata((vac_coords[:, 0], vac_coords[:, 1]),
    #     #                 vac_coords[:, 2],
    #     #                 (xnew, ynew),
    #     #                 method='linear')
    #
    #     self.wavefront = griddata((vac_coords[:, 0], vac_coords[:, 1]),
    #                     np.append(0, self._wft),
    #                     (xnew, ynew),
    #                     method='linear')

    def _initModalBases(self, nbOfModes=100):
        diam = 1
        radial_cutoff = False

        self.smallGrid = hcipy.make_pupil_grid(self.cfg.params.asym_nsteps - 1)
        self.M2C_small = hcipy.make_zernike_basis(nbOfModes, diam,
                                                   self.smallGrid,
                                                  4,
                                                  radial_cutoff=radial_cutoff)
        self.M2C_small = psi_utils.reorthonormalize(self.M2C_small, self._small_aperture.flatten())
        self.C2M_small = hcipy.inverse_tikhonov(self.M2C_small.transformation_matrix, 1e-3)

        self.M2C_large = hcipy.make_zernike_basis(nbOfModes, diam,
                                                  self.inst.pupilGrid, 4,
                                                  radial_cutoff=radial_cutoff)
        # binary_aperture = np.copy(self.inst.aperture)
        # binary_aperture[self.inst.aperture >=0.5] = 1
        # binary_aperture[self.inst.aperture < 0.5] = 0
        self.M2C_large = psi_utils.reorthonormalize(self.M2C_large, self.inst.aperture)
        self.C2M_large =hcipy.inverse_tikhonov(self.M2C_large.transformation_matrix, 1e-3)

    def _modalFilteringOnEP(self, ncpa_estimate):
        '''
            Modal projection (/filtering) on entrance pupil grid

            TODO: uniformize method namings
        '''
        ncpa_modes      = self.C2M_large.dot(ncpa_estimate.flatten() * self.inst.aperture.flatten())
        ncpa_estimate  = self.M2C_large.transformation_matrix.dot(ncpa_modes)

        return ncpa_estimate, ncpa_modes

    def _projectOnModalBasis(self, ncpa_estimate, proj_mask):
        '''
            TODO: clarify method naming
        '''
        ncpa_modes      = self.C2M_small.dot(ncpa_estimate.flatten() * proj_mask.flatten())
        ncpa_estimate  = self.M2C_large.transformation_matrix.dot(ncpa_modes)

        return ncpa_estimate, ncpa_modes

    def next(self, display=True, check=False, gain=0.5, leak=0.9, integrator=True):
        nbOfSeconds = 1/self.cfg.params.psi_framerate
        science_images_buffer = self.inst.grabScienceImages(nbOfSeconds)

        self.science_image = science_images_buffer.mean(0)
        self.computeWavefront(self.science_image)
        self._ncpa_estimate, _ = self._projectOnModalBasis(self.wavefront, self._small_aperture)

        self._ncpa_command = - gain * self._ncpa_estimate

        self.inst.setNcpaCorrection(self._ncpa_command, integrator=integrator, leak=leak)
        self.iter +=1


    def evaluateSensorEstimate(self, verbose=True):
        res_ncpa_qs = self.inst.phase_ncpa + self.inst.phase_ncpa_correction
        res_ncpa_all = self.inst.phase_ncpa + self.inst.phase_wv + \
            self.inst.phase_ncpa_correction
        # Residual over the average WV over one iteration (/ detector integration)
        res_ncpa_all_bis = self.inst.phase_ncpa + self.inst.phase_wv_integrated + \
            self.inst.phase_ncpa_correction

        conv2nm = self.inst.wavelength / (2 * np.pi) * 1e9

        # TODO: For asymmetric Lyot, maybe we want to use lyot stop instead of EP
        rms_res_qs = np.std(res_ncpa_qs[self.inst.aperture>=0.5]) * conv2nm
        rms_res_all = np.std(res_ncpa_all[self.inst.aperture>=0.5]) * conv2nm
        rms_res_all_bis = np.std(res_ncpa_all_bis[self.inst.aperture>=0.5]) * conv2nm

        # rms residual errors
        tmp, _ = self._modalFilteringOnEP(res_ncpa_qs)
        rms_res_qs_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
        tmp, _ = self._modalFilteringOnEP(res_ncpa_all)
        rms_res_all_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
        tmp, _ = self._modalFilteringOnEP(res_ncpa_all_bis)
        rms_res_all_bis_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

        # rms WV perturbations and
        tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv)
        rms_wv = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
        tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv_integrated)
        rms_wv_integrated = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

        tmp, _ = self._modalFilteringOnEP(self.inst.phase_ncpa_correction)
        rms_corr = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

        if verbose:
            self.logger.info('\n#{0} : Res [QS, QS+WV,  QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
                format(self.iter, rms_res_qs, rms_res_all, rms_res_all_bis))
            self.logger.info('#{0} : Res. modal sfilt. [QS, QS+WV, QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
                format(self.iter, rms_res_qs_filt, rms_res_all_filt, rms_res_all_bis_filt))
            self.logger.info('#{0} : Input WV_f rms (last, integrated)  = ({1:.0f}, {2:.0f})'.format(self.iter, rms_wv, rms_wv_integrated))
            self.logger.info('#{0} : Sensor correction rms = {1:.0f}'.format(self.iter, rms_corr))


        loop_stat = [self.iter]
        loop_stat.append(rms_res_all_filt)
        loop_stat.append(rms_res_qs_filt)
        loop_stat.append(rms_res_all)
        loop_stat.append(rms_res_qs)
        # [01/07/2022] : added 01/07/2022
        loop_stat.append(rms_wv_integrated)  # input WV average over 1/psi_framertae -- on the modes
        loop_stat.append(rms_res_all_bis_filt)    # rms all considering the average WV and not the instantaneoius
        # loop_stat.append(rms_res_static_NCPA_filt)  # long-term average of the correction compared to the QS part
        self._loop_stats.append(loop_stat)
