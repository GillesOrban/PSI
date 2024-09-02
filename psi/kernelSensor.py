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
import matplotlib.animation as animation
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
from psi.abstract_sensor import AbstractSensor
from .configParser import loadConfiguration
from .instruments import  CompassSimInstrument, DemoCompassSimInstrument, HcipySimInstrument
from .helperFunctions import LazyLogger, timeit, build_directory_name, \
    copy_cfgFileToDir, dump_config_to_text_file

from astropy.visualization import imshow_norm,\
    SqrtStretch, MinMaxInterval, PercentileInterval, \
    LinearStretch, SinhStretch, LogStretch, ManualInterval


# from config.config_metis_compass import conf

# sys.path.append('/Users/orban/Projects/METIS/4.PSI/legacy_TestArea/')
print(hcipy.__file__)
print(hcipy.__version__)




sys.path.append('/Users/orban/github/')

from xara import xara
from xaosim import xaosim as xs
from scipy.interpolate import griddata


class KernelSensor(AbstractSensor):
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
        super().__init__()
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
            Setup the wavefront sensor based on the configuration
        '''
        # Build instrument object 'inst'
        self.logger.info('Initialize the instrument object & building the optical model')
        # self.inst = getattr(instruments,
        #                     self.cfg.params.instrument)(self.cfg.params)
        self.inst = eval(self.cfg.params.instrument)(self.cfg.params)
        # importlib.import_module
        self.inst.build_optical_model()
        if hasattr(self.cfg.params, 'tel_diam'):
            self._Dtel = self.cfg.params.tel_diam
        else:
            self.logger.warning('Telescope diameter not given in config. Setting to 1')
            self._Dtel = 1 #self.cfg.params.asym_telDiam
        self._pscale = (self.cfg.params.wavelength / self._Dtel) * 206264.8 / self.cfg.params.det_res * 1e3 # [mas / px]

        #pscale = 4.91168055, #5.32359, #5.47,                      # pixel scale in mas/pix

        # Build or load the Kernel model
        if self.cfg.params.asym_model_fname is None or '':
            self.logger.info('Builing KPI model')
            now = datetime.datetime.now()
            timetag = now.strftime("%Y-%m-%dT%H:%M:%S")
            self.cfg.params.asym_model_fname = 'KPI_{0}.fits.gz'.format(timetag)
            self._buildModelKPI(fname=self.cfg.params.asym_model_fname)
        else:
            self.logger.info('Loading KPI model : '
                             '{0}'.format(self.cfg.params.asym_model_fname))
        
        self._loadKPO(fname_model=self.cfg.params.asym_model_fname)


        # # Build focal plane filter for PSI
        self.filter_fp = psi_utils.makeFilters(
            self.inst.focalGrid,
            "back_prop",
            sigma=self.cfg.params.psi_filt_sigma,
            lD=self.cfg.params.psi_filt_radius * 2,
            ravel=False
            )


        # Init logging buffer
        self._ncpa_correction_long_term = 0
        self.iter = 0 # iteration index
        self._loop_stats = []

        # -- Plotting & saving results
        # self.fig = plt.figure(figsize=(9, 3))
        if self.cfg.params.save_loop_statistics:
            if self.cfg.params.save_dirname is None or self.cfg.params.save_dirname=='': 
                self._directory = build_directory_name(self._config_file,
                                                self.cfg.params.save_basedir)
            else:
                self._directory = self.cfg.params.save_basedir + self.cfg.params.save_dirname

            if not os.path.exists(self._directory):
                os.makedirs(self._directory)

            # copy initial config file to directory
            copy_cfgFileToDir(self._directory, self._config_file)

            # copy current configuration to text file
            dump_config_to_text_file(self._directory + '/config/' + 'current_config.txt',
                                     self.cfg.params)

            if self.cfg.params.save_phase_screens:
                self._directory_phase = self._directory + 'residualNCPA/'
                os.mkdir(self._directory_phase)

            self.logger.info('Results will be stored in '
                             '{0}'.format(self._directory))

    # def _save_loop_stats(self):
    #     '''
    #         Saving loop statistics to file

    #         16/06/2023  copied from PsiSensor
    #     '''
    #     data = np.array(self._loop_stats)
    #     np.savetxt(os.path.join(self._directory, 'loopStats.csv'),
    #               data,
    #               header ='units are nm \n it \t wfe_all_f \t wfe_qs_f \t wfe_all \t wfe_qs \t input_wv_avg \t wfe_all_f_avg \t wfe_static',
    #               fmt=['%i' , '%f', '%f', '%f', '%f', '%f', '%f', '%f'],
    #               delimiter= '\t')
        

    def _buildModelKPI(self, fname="scexao_asym.fits.gz", verbose=False):
        '''
            TODO KPI model with finer pupil image ? -> impact on accuracy and matrix sizes ?
            TODO Adapt bmax in very low SNR cases ? -- check 1.1*Dtel is optimum (or better 1)
        '''
        # Create model
        pscale = self._Dtel / self.inst._size_pupil_grid
        step= self._Dtel / self.cfg.params.asym_nsteps # [meter] # 0.3
        # Ensure that the step is an even integer number of pixels (see 'create_discret_model' help)
        step_pix = np.round(step / pscale)
        if np.mod(step_pix, 2):
            step_pix -=1
        self._step = step_pix * pscale # force step to be an integer nb of pixels
        self._nsteps = int(self._Dtel / self._step)
        if np.mod(self._nsteps, 2) == 0:
            self._nsteps += 1
        self.logger.warn('Final number of subap on diam is {0} with a pitch of {1:.3f}'.format(self._nsteps, self._step))
        tmin = self.cfg.params.asym_tmin # 0.1 # cut-off value for transmissive or not
        model  = xara.core.create_discrete_model(self.inst.aperture.shaped,
                                                 pscale,
                                                 self._step,
                                                 binary=False,
                                                 tmin=tmin)
        # bmax : safeguard that tells the KPO data structure creation function to
        #         discard any resulting baseline that would be greater than the provided value
        kpi = xara.KPI(array=model, bmax=self._Dtel) # TBC -> bmax needed ??

        # Save model
        self.logger.info('Saving KPI model to {0}'.format(fname))
        kpi.save_to_file(fname)

        # Display
        if verbose:
            kpi.plot_pupil_and_uv()
    

    def _loadKPO(self, fname_model="scexao_asym.fits.gz", neig=None):
        '''
            fname_model : .fits or .fits.gz

            TFM : Transfer phase Matrix = R^-1 A
        '''
        self.kpo    = xara.KPO(fname=fname_model) # load the KPO structure
        U, S, Vt = np.linalg.svd(self.kpo.kpi.TFM, full_matrices=0)

        # filter some of the low singular values
        if neig is None:
            #neig = int(0.4*len(S))     # 200 nomber of singular values to keep (max=509 for this model)
            ss = S / S.max()
            idx = np.argwhere(ss < 1e-3)
            if len(idx) == 0:
                neig=len(ss)
            else:
                neig = idx[0][0]
        else:
            if neig > len(S):
                neig = len(S)
        self.logger.info('Keeping {0:.0f}% of the eigenvalues : {1}'.format(np.round(neig/len(S)*100),
                                                                        neig))
        Sinv = 1 / S
        Sinv[neig:] = 0.0

        # computation of the pseudo inverse
        self.phase_tf_inv = Vt.T.dot(np.diag(Sinv)).dot(U.T)

        # define small aperture 
        nbs = self._nsteps #self.cfg.params.asym_nsteps
        asym_telDiam = self._Dtel #self.cfg.params.asym_telDiam
        # step_size = asym_telDiam / nbs

        self._small_aperture = np.zeros((nbs, nbs))
        vac_coords = self.kpo.kpi.VAC
        coords = np.array(vac_coords[:,0:2] / self._step + \
                    (asym_telDiam/2) / self._step, dtype='int')
        self._small_aperture[list(coords[:,1]), list(coords[:,0])] = vac_coords[:,2]
        # self._small_aperture[self._small_aperture!=0]=1

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
        asym_telDiam = self._Dtel #self.cfg.params.asym_telDiam
        asym_nsteps = self._nsteps #self.cfg.params.asym_nsteps

        # step_size = asym_telDiam / asym_nsteps

        coords = np.array(vac_coords[:,0:2] / self._step + \
                         (asym_telDiam/2) / self._step, dtype='int')

        nbs=asym_nsteps
        # if np.mod(nbs, 2):
        #     '''if odd'''
        #     phase = np.zeros((nbs, nbs))
        #     phase[list(coords[:,1]), list(coords[:,0])] = np.append(0, self._wft)#vac_coords[:,2]
        # else:
        # phase = np.zeros((nbs-1, nbs-1))
        # phase[list(coords[:,1]-1), list(coords[:,0]-1)] = np.append(0, self._wft)#vac_coords[:,2]
        phase = np.zeros((nbs, nbs))
        phase[list(coords[:,1]), list(coords[:,0])] = np.append(0, self._wft)



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

    def _initModalBases(self, nbOfModes=100, nmode_shift=3, reortho=True):
        '''
        FIXME re-orthonormalization leads to NaNs for the small aperture
            (because odd grid ??)
            check the basis -> impact of diam=1

        PARAMETERS
            nmode_shift  :    int
                by default skip the first 3 modes, ie piston, tip and tilt
        '''

        if self.cfg.params.modal_basis == 'Dfull_modes':
            # -- for instrument pupil
            # assume modal basis and inst pupil have same dimensions
            self.logger.info('Using the SCAO Dfull basis {0} with {1} modes'.format(os.path.basename(self.cfg.params.f_modal_basis),
                                                                                    nbOfModes))

            renormalize_modal_basis = 1e6 # from meters to microns
            scao_modes_raw = fits.getdata(self.cfg.params.f_modal_basis) * renormalize_modal_basis

            nmode_shift -= 1 # assume that the scao basis does not include piston, unlike HCIPy
            assert scao_modes_raw.shape[0] >= nbOfModes + nmode_shift

            dim = scao_modes_raw.shape[1]
            assert dim == self.inst.pupilGrid.dims[0], 'Pupil grid and modal basis have different dimensions'
            scao_basis = hcipy.ModeBasis(np.reshape(scao_modes_raw[:nbOfModes+nmode_shift],
                                                    (nbOfModes+nmode_shift, dim * dim)).T, 
                                        self.inst.pupilGrid)
            M2C = scao_basis.transformation_matrix[:, nmode_shift:]
            C2M = hcipy.inverse_tikhonov(scao_basis.transformation_matrix,
                                               1e-3)[nmode_shift:,:]

            self.M2C = M2C
            self.C2M = C2M

            # -- for kernel discretized pupil
            self.smallGrid = hcipy.make_pupil_grid(self.cfg.params.asym_nsteps)
            # scao_modes_raw_small = psi_utils.resize_img(scao_modes_raw, self.cfg.params.asym_nsteps)
            scao_modes_raw_small = psi_utils.resize_img(scao_modes_raw, self._nsteps)

            dim = scao_modes_raw_small.shape[1]
            scao_basis_small = hcipy.ModeBasis(np.reshape(scao_modes_raw_small[:nbOfModes+nmode_shift],
                                                          (nbOfModes+nmode_shift, dim * dim)).T, 
                                              self.smallGrid)
            M2C_small = scao_basis_small.transformation_matrix[:, nmode_shift:]
            C2M_small = hcipy.inverse_tikhonov(scao_basis_small.transformation_matrix,
                                               1e-3)[nmode_shift:,:]
            
            if self.cfg.params.inst_mode == 'SPP':
                self.logger.warn('SPP mask not used to compute M2C and C2M / M2C_small and C2M_small')

            self.M2C_matrix_small = M2C_small
            self.C2M_small = C2M_small

        else:
            #----------------------------------------#
            #-- For kernel discretized pupil
            self.smallGrid = hcipy.make_pupil_grid(self.cfg.params.asym_nsteps)

            M2C, C2M = psi_utils.makeModalBasis(self.smallGrid,
                                                nbOfModes,
                                                nmode_shift,
                                                reortho=reortho,
                                                aperture=self._small_aperture.flatten(),
                                                basis_name=self.cfg.params.modal_basis)
            self.M2C_matrix_small = M2C
            self.C2M_small = C2M

            if self.cfg.params.inst_mode == 'SPP':
                self.logger.warn('SPP mask used to reorthogonalize M2C_small and C2M_small but not to compute M2C and C2M')
            #----------------------------------------#
            #-- For instrument pupil
            self.logger.info('Initializing modal basis with {0} modes'.format(nbOfModes))
            
            aper=None
            if reortho:
                if self.cfg.params.pupil == 'ELT':
                    grid_diam = self.inst.pupilGrid.delta[0] * self.inst.pupilGrid.dims[0]
                    aper = hcipy.aperture.make_circular_aperture(0.98 * grid_diam)(self.inst.pupilGrid)
                    aper -= hcipy.aperture.make_circular_aperture(0.25 * grid_diam)(self.inst.pupilGrid)
                    tmpGrid = self.inst.pupilGrid.copy().scale(1/grid_diam)
                    aper *= self.inst._asym_mask(tmpGrid)
                else:
                    aper = self.inst.aperture
            M2C, C2M = psi_utils.makeModalBasis(self.inst.pupilGrid,
                                                nbOfModes,
                                                nmode_shift,
                                                reortho=reortho,
                                                aperture=aper,
                                                basis_name=self.cfg.params.modal_basis)
            # self.M2C_basis = basis
            self.M2C = M2C
            self.C2M = C2M

    # def _modalFilteringOnEP(self, ncpa_estimate):
    #     '''
    #         Modal projection (/filtering) on entrance pupil grid

    #         TODO: uniformize method namings
    #     '''
    #     ncpa_modes      = self.C2M_large.dot(ncpa_estimate.flatten() * self.inst.aperture.flatten())
    #     # ncpa_estimate  = self.M2C_large.transformation_matrix.dot(ncpa_modes)
    #     ncpa_estimate  = self.M2C_matrix_large.dot(ncpa_modes)

    #     return ncpa_estimate, ncpa_modes

    def _projectOnModalBasis(self, ncpa_estimate, proj_mask, modal_gains=1):
        '''
            TODO: clarify method naming
        '''
        ncpa_modes      = self.C2M_small.dot(ncpa_estimate.flatten() * proj_mask.flatten())
        # ncpa_estimate  = self.M2C_matrix_large.dot(modal_gains * ncpa_modes)
        ncpa_estimate  = self.M2C.dot(modal_gains * ncpa_modes)


        return ncpa_estimate, ncpa_modes

    def next(self, display=True, check=False, leak=1, gains=[None,None],
             integrator=True, modal_gains=1):
        # TODO replace psi_framerate by kernel_framerate
        nbOfSeconds = 1/self.cfg.params.framerate
        science_images_buffer = self.inst.grabScienceImages(nbOfSeconds)
        self.science_image = science_images_buffer.mean(0)

        self.inst.synchronizeBuffers(None, None)

        self.computeWavefront(self.science_image * self.filter_fp)
        self._ncpa_estimate, _ = self._projectOnModalBasis(self.wavefront,
                                                           self._small_aperture,
                                                           modal_gains)
        # Remove piston
        self._ncpa_estimate -= np.mean((self._ncpa_estimate * 
                                       self.inst.aperture)[self.inst.aperture !=0])
        if gains == [None, None]:
            gain_I = self.cfg.params.gain_I
            gain_P = self.cfg.params.gain_P
        else:
            gain_I = gains[0]
            gain_P = gains[1]
        
        self._ncpa_command = - gain_I * self._ncpa_estimate
        
        self.inst.setNcpaCorrection(self._ncpa_command,
                                    phase_prop=gain_P * self._ncpa_estimate,
                                    integrator=integrator, leak=leak)
        self.iter +=1

        if display:
            self.show()


    # def show(self):
    #     '''
    #     TODO tidy up and find solution for live display and video saving
    #     '''
    #     # plt.clf()

    #     # plt.subplot(141)
    #     # imshow_norm(img.mean(0), stretch=LogStretch())
    #     ax1 = plt.subplot(141)
    #     im1, _= imshow_norm(self.science_image, stretch=LogStretch(), ax=ax1)
    #     # plt.subplot(142)
    #     # inter = PercentileInterval(99.5)
    #     vmin = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 1)
    #     vmax = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 99)
    #     inter = ManualInterval(vmin, vmax)
    #     ax2 = plt.subplot(142)
    #     im2, _=imshow_norm((self.inst.phase_wv + self.inst.phase_ncpa).shaped,
    #                        interval=inter, ax=ax2)
    #     # plt.subplot(143)
    #     # imshow_norm(self._ncpa_estimate.reshape((256, 256)) * self.inst.aperture.shaped, interval=inter)
    #     ax3 = plt.subplot(143)
    #     _dim = self.inst.pupilGrid.shape[0]
    #     im3, _=imshow_norm(-self.inst.phase_ncpa_correction.reshape((_dim, _dim)) * \
    #                        self.inst.aperture.shaped,
    #                        interval=inter, ax=ax3)
    #     # plt.subplot(144)
    #     ax4 = plt.subplot(144)
    #     im4, _=imshow_norm(self.inst.aperture.shaped *
    #                        (self.inst.phase_wv + self.inst.phase_ncpa +
    #                        self.inst.phase_ncpa_correction).shaped,
    #                        interval=inter, ax=ax4)

    #     ax1.set_axis_off()
    #     ax2.set_axis_off()
    #     ax3.set_axis_off()
    #     ax4.set_axis_off()
    #     ax2.set_title('static NCPA + WV')
    #     ax3.set_title('NCPA correction')
    #     ax4.set_title('Residuals')
    #     plt.tight_layout()
    #     # plt.draw()
    #     # plt.pause(0.01)
    #     # time.sleep(0.1)
    #     self._ims.append([im1, im2, im3, im4])

    def loop(self, **kwargs):
        self._ims = []  # Matplotlib Artist list 
        db_logger = kwargs.get('db_logger', None)
        if 'db_logger' in kwargs.keys():
            kwargs.pop('db_logger')
        for i in range(self.cfg.params.nb_iter):
            self.next(**kwargs)
            self.evaluateSensorEstimate(db_logger=db_logger)
            # if self.cfg.params.save_phase_screens:
            #     self._store_phase_screens_to_file(self.iter)

        if self.cfg.params.save_loop_statistics:
            self._save_loop_stats()

    def save_video(self, fname='toto.mp4', fps=0.5):
        fig=plt.gcf()
        ani = animation.ArtistAnimation(fig, self._ims, interval=500,
                                        blit=False, repeat=False)
        FFwriter = animation.FFMpegWriter(fps=fps)
        ani.save(fname, writer=FFwriter)

    # def evaluateSensorEstimate(self, verbose=True):
    #     res_ncpa_qs = self.inst.phase_ncpa + self.inst.phase_ncpa_correction
    #     res_ncpa_all = self.inst.phase_ncpa + self.inst.phase_wv + \
    #         self.inst.phase_ncpa_correction
    #     # Residual over the average WV over one iteration (/ detector integration)
    #     res_ncpa_all_bis = self.inst.phase_ncpa + self.inst.phase_wv_integrated + \
    #         self.inst.phase_ncpa_correction

    #     conv2nm = self.inst.wavelength / (2 * np.pi) * 1e9

    #     # TODO: For asymmetric Lyot, maybe we want to use lyot stop instead of EP
    #     rms_res_qs = np.std(res_ncpa_qs[self.inst.aperture>=0.5]) * conv2nm
    #     rms_res_all = np.std(res_ncpa_all[self.inst.aperture>=0.5]) * conv2nm
    #     rms_res_all_bis = np.std(res_ncpa_all_bis[self.inst.aperture>=0.5]) * conv2nm

    #     # rms residual errors
    #     tmp, _ = self._modalFilteringOnEP(res_ncpa_qs)
    #     rms_res_qs_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #     tmp, _ = self._modalFilteringOnEP(res_ncpa_all)
    #     rms_res_all_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #     tmp, _ = self._modalFilteringOnEP(res_ncpa_all_bis)
    #     rms_res_all_bis_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     # rms WV perturbations and
    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv)
    #     rms_wv = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv_integrated)
    #     rms_wv_integrated = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_ncpa_correction)
    #     rms_corr = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     tmp, _ = self._projectOnModalBasis(res_static_ncpa_qs)
    #     rms_res_static_NCPA_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm


    #     if verbose:
    #         self.logger.info('\n#{0} : Res [QS, QS+WV,  QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
    #             format(self.iter, rms_res_qs, rms_res_all, rms_res_all_bis))
    #         self.logger.info('#{0} : Res. modal sfilt. [QS, QS+WV, QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
    #             format(self.iter, rms_res_qs_filt, rms_res_all_filt, rms_res_all_bis_filt))
    #         self.logger.info('#{0} : Input WV_f rms (last, integrated)  = ({1:.0f}, {2:.0f})'.format(self.iter, rms_wv, rms_wv_integrated))
    #         self.logger.info('#{0} : Sensor correction rms = {1:.0f}'.format(self.iter, rms_corr))


    #     loop_stat = [self.iter]
    #     loop_stat.append(rms_res_all_filt)
    #     loop_stat.append(rms_res_qs_filt)
    #     loop_stat.append(rms_res_all)
    #     loop_stat.append(rms_res_qs)
    #     # [01/07/2022] : added 01/07/2022
    #     loop_stat.append(rms_wv_integrated)  # input WV average over 1/psi_framertae -- on the modes
    #     loop_stat.append(rms_res_all_bis_filt)    # rms all considering the average WV and not the instantaneoius
    #     loop_stat.append(rms_res_static_NCPA_filt)  # long-term average of the correction compared to the QS part
    #     self._loop_stats.append(loop_stat)

    # def evaluateSensorEstimate(self, verbose=True):
    #     '''
    #         Compute the rms errors made on quasi-static NCPA and on water vapour seeing.

    #         /!\ Only valid for a `CompassSimInstrument` and `DemoCompassSimInstrument`

    #         TODO make it generic to any instruments
    #     '''
    #     res_ncpa_qs = self.inst.phase_ncpa + self.inst.phase_ncpa_correction
    #     res_ncpa_all = self.inst.phase_ncpa + self.inst.phase_wv + \
    #         self.inst.phase_ncpa_correction
    #     # 2022-06-2x ...
    #     if self.iter == 0:
    #         res_static_ncpa_qs = self.inst.phase_ncpa
    #     else:
    #         # tmp_avg = np.mean(self.inst.phase_ncpa_correction[self.inst.aperture>=0.5])
    #         self._ncpa_correction_long_term += self.inst.phase_ncpa_correction #- tmp_avg)
    #         # self._ncpa_correction_long_term /= self.iter

    #         res_static_ncpa_qs = self.inst.phase_ncpa + (self._ncpa_correction_long_term / self.iter)

    #     # 2022-07-01 -- metric with the average WV over one iteration
    #     res_ncpa_all_bis = self.inst.phase_ncpa + self.inst.phase_wv_integrated + \
    #         self.inst.phase_ncpa_correction

    #     conv2nm = self.inst.wavelength / (2 * np.pi) * 1e9
    #     # rms_input_qs = np.std(self.inst.phase_ncpa[self.inst.aperture==1]) * conv2nm
    #     # rms_input_all = np.std((self.inst.phase_ncpa + \
    #     #                         self.inst.phase_wv)[self.inst.aperture==1]) * conv2nm
    #     rms_res_qs = np.std(res_ncpa_qs[self.inst.aperture>=0.5]) * conv2nm
    #     rms_res_all = np.std(res_ncpa_all[self.inst.aperture>=0.5]) * conv2nm
    #     rms_res_all_bis = np.std(res_ncpa_all_bis[self.inst.aperture>=0.5]) * conv2nm


    #     if self.cfg.params.psi_correction_mode is not 'all':
    #         tmp, _ = self._modalFilteringOnEP(res_ncpa_qs)
    #         rms_res_qs_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #         tmp, _ = self._modalFilteringOnEP(res_ncpa_all)
    #         rms_res_all_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #         tmp, _ = self._modalFilteringOnEP(res_ncpa_all_bis)
    #         rms_res_all_bis_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #     else:
    #         rms_res_qs_filt = rms_res_qs
    #         rms_res_all_filt = rms_res_all

    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv)
    #     rms_wv = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv_integrated)
    #     rms_wv_integrated = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_ncpa_correction)
    #     rms_corr = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     tmp, _ = self._modalFilteringOnEP(res_static_ncpa_qs)
    #     rms_res_static_NCPA_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     if verbose:
    #         self.logger.info('#{0} : Res [QS, QS+WV,  QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
    #             format(self.iter, rms_res_qs, rms_res_all, rms_res_all_bis))
    #         self.logger.info('#{0} : Res. filt. [QS, QS+WV, QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
    #             format(self.iter, rms_res_qs_filt, rms_res_all_filt, rms_res_all_bis_filt))
    #         self.logger.info('#{0} : input WV_f rms (last, integrated)  = ({1:.0f}, {2:.0f})'.format(self.iter, rms_wv, rms_wv_integrated))
    #         self.logger.info('#{0} : PSI correction rms = {1:.0f}'.format(self.iter, rms_corr))
    #         self.logger.info('#{0} : Long-term (static) residual rms = {1:.0f}'.format(self.iter, rms_res_static_NCPA_filt))


    #     loop_stat = [self.iter]
    #     loop_stat.append(rms_res_all_filt)
    #     loop_stat.append(rms_res_qs_filt)
    #     loop_stat.append(rms_res_all)
    #     loop_stat.append(rms_res_qs)
    #     # [01/07/2022] : added 01/07/2022
    #     loop_stat.append(rms_wv_integrated)  # input WV average over 1/psi_framertae -- on the modes
    #     loop_stat.append(rms_res_all_bis_filt)    # rms all considering the average WV and not the instantaneoius
    #     loop_stat.append(rms_res_static_NCPA_filt)  # long-term average of the correction compared to the QS part
    #     self._loop_stats.append(loop_stat)