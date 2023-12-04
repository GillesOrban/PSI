import traceback
import sys
import os
# sys.path.append('/Users/orban/Projects/METIS/4.PSI/psi_github/')
from .helperFunctions import LazyLogger
from .psi_utils.photometry_definition import PHOT
from .psi_utils.apertures import mask_asym_baseline, mask_asym_two, mask_asym_two_lyot
import os

class ConfigurationError(Exception):
    pass

class Parameters(object):
    """
    TODO uniform inst_mode names
    TODO complete the description of the config parameters
    TODO check which sensor is used : kernel, psi. And check relevant parameters

    Args:
        filename (string): The name of the configuration file



    Description of the config parameters
        ========================      ===================
        **Required Parameter**        **Description**
        ------------------------      -------------------
        ``npupil``                    int: number of pixels of the pupil
        ``det_size``                  -
        ``det_res``                   -
        ``instrument``                -
        ``inst_mode``                 -
        ``vc_charge``                 -
        ``vc_vector``                 -
        ``f_aperture``                -
        ``f_lyot_stop``               -
        ``dit``                       -
        ``ao_framerate``              -
        ``ao_frame_decimation``       -
        ``psi_framerate``             -
        ``psi_nb_iter``               -
        ``psi_correction_mode``       -
        ``psi_nb_modes``              -
        ``psi_start_mode_idx``        -
        ``ncpa_expected_rms``         -
        ``save_loop_statistics``      -
        ``save_phases_screens``       -
        ``save_basedir``              -
        ========================      ===================

        ======================      ===================
        **Optional Parameter**      **Description**
        ----------------------      -------------------
        ``noise``                   -
        ``mag``                     -
        ``wavelength``              - req ?
        ``flux_zpt``                -
        ``flux_bckg``               -
        ``bandwidth``               - 
        ``ncpa_dynamic``            -
        ``ncpa_sampling``           -
        ``ncpa_scaling``            -
        ``ncpa_folder``             -
        ``ncpa_prefix``             -
        ``turb_folder``             -
        ``turb_prefix_rp``          -
        ``turb_prefix_wf``          -
        ``turb_suffix``             -
        ``wv``                      -
        ``wv_folder``               -
        ``wv_cube_name``            -
        ``wv_sampling``             -
        ``wv_scaling``              -
        ======================      ===================
    """

    def __init__(self, filename, logger=LazyLogger('Params')):
        self.filename = filename
        self.logger = logger

    def readfile(self):
        '''
        Read configuration file -> create `conf` namespace
        '''

        #Exec the config file, which should contain a dict ``simConfiguration``
        try:
            with open(self.filename) as file_:
                exec(file_.read(), globals())
        except:
            traceback.print_exc()
            raise ConfigurationError(
                    "Error loading config file: {}".format(self.filename))

        # self.configDict = simConfiguration
        self.params = conf

    def check_parameters(self):
        '''
            Performs a number of sanity checks on the configuration parameters
        '''
        if hasattr(self.params, 'band'):
            # if parameters as the parameter band defined, 
            # read the photometric definition from the dict
            self.params.wavelength = PHOT[self.params.band]['lam']
            self.params.flux_zpt = PHOT[self.params.band]['flux_star']
            self.params.flux_bckg = PHOT[self.params.band]['flux_bckg']
            self.params.pscale = PHOT[self.params.band]['pscale']

        if not(hasattr(self.params, 'telescope_diameter')):
            # TODO see how telescope_diameter can be use by instrument HCIpySimulator
            # Assume ELT by default
            self.params.telescope_diameter = 36.905
        if not(hasattr(self.params, 'pscale')):
            # default pixel scale given by METIS_L
            self.params.pscale = 5.47
        if not(hasattr(self.params, 'det_res')):
            self.params.det_res = None

        # TODO check consistency between Lyot stop and band
        # check, based on the band, that the zeropoint is according to defined 'constants'
        #  otherwise print a 'warning'
        if self.params.inst_mode == 'CVC':
            # if self.params.band == 'L':
            if os.path.basename(self.params.f_lyot_stop)[0:8] != 'ls_CVC_L':
                self.logger.warn(('Lyot stop fname does not seem to match for {0}'
                             'Please check the filename').format(self.params.inst_mode))
            # if self.params.band == 'N':
            #     if os.path.basename(self.params.f_lyot_stop)[0:8] != 'ls_CVC_N':
            #         self.logger(('Lyot stop fname does not seem to match for {0}.'
            #                      ' Please check the filename').format(self.params.inst_mode))
        elif self.params.inst_mode == 'RAVC':
            # if self.params.band == 'L':
            if os.path.basename(self.params.f_lyot_stop)[0:9] != 'ls_RAVC':
                self.logger.warn(('Lyot stop fname does not seem to match for {0}'
                             '  Please check the filename').format(self.params.inst_mode))

            # if self.params.band == 'N':
            #     if os.path.basename(self.params.f_lyot_stop)[0:8] != 'ls_RAVC_N':
            #         self.logger(('Lyot stop fname does not seem to match for {0}.'
            #                      ' Please check the filename').format(self.params.inst_mode))

        if hasattr(self.params, 'psi_filt_radius'):
            assert self.params.det_size >= self.params.psi_filt_radius

        if hasattr(self.params, 'psi_correction_mode'):
            # PSI correction mode checks
            if self.params.psi_correction_mode == 'zern':
                if hasattr(self.params, 'psi_nb_modes') is False:
                    default_nb_modes = 20
                    self.logger.warn('Setting default psi_nb_modes to {0}'.\
                        format(default_nb_modes))
                    self.params.psi_nb_modes = default_nb_modes
                if hasattr(self.params, 'psi_start_mode_idx') is False:
                    default_start_idx = 4
                    self.logger.warn('Setting default psi_start_mode_idx to {0}'.\
                        format(default_start_idx))
                    self.params.psi_start_mode_idx = default_start_idx

        if not(hasattr(self.params, 'gain_I')):
            self.params.gain_I = 0.4
            self.params.gain_P = 0
        if not(hasattr(self.params, 'gain_P')):
            self.params.gain_P = 0
        # Check if using ``CompassSimInstrument``
        if self.params.instrument == 'CompassSimInstrument':
            assert self.params.npupil == 256, 'Array size for CompassSimInstrument needs to be 256'
            if os.path.isfile(self.params.f_aperture) is False:
                self.logger.error('No aperture file, cannot proceed')
                raise ConfigurationError("No aperture file")
        # pass

        # Saving
        if self.params.save_phase_screens and not(self.params.save_loop_statistics):
            self.logger.warn('Setting save results to True')
            self.params.save_results = True

        # Set default attributes if not already defined 
        # Check is asymmetric pupil/lyot stop (kernel WFS). If not, set to None
        if hasattr(self.params, 'asym_stop') is False:
            self.params.asym_stop = None

        if self.params.asym_stop == True:
            self.params.asym_nsteps <= 2 * self.params.det_size  # pitch cannot be finer than set by field-of-view

        if hasattr(self.params, 'bandwidth') is False:
            self.params.bandwidth = 0


        if self.params.instrument == 'HcipySimInstrument':
            if self.params.pupil == 'ELT':
                self.params.tel_diam = 40
            elif self.params.pupil == 'ERIS':
                self.params.tel_diam = 8.1196  # UT4
            elif self.params.pupil == 'CIRC':
                self.params.tel_diam = 8
            else:
                raise ConfigurationError('Pupil does not exist')


    def compute_parameters(self):
        '''
        Compute some parameters based on the configuration file parameters
        '''
        if self.params.det_res is None:
            self.params.det_res = self.params.wavelength /\
                  self.params.telescope_diameter * 206264.8 /\
                  (self.params.pscale * 1e-3)
            self.logger.info('Detector resolution is '
                             '{0:.2f} px/(lbda/D)'.format(self.params.det_res))

        self.params.nb_ao_frames_per_science = int(self.params.dit / \
            (self.params.ao_frame_decimation / self.params.ao_framerate))

        # if Simulation
        self.params.num_photons = self.params.dit * self.params.flux_zpt * \
            10**(-0.4 * self.params.mag)

        self.params.num_photons_bkg = self.params.dit * self.params.flux_bckg

        if self.params.asym_stop is True:
            if self.params.instrument == 'HcipySimInstrument':
                self.params.asym_width *= self.params.tel_diam
            if hasattr(self.params, 'asym_mask_option') is False:
                self.params.asym_mask = mask_asym_baseline(self.params.asym_width,
                                                           self.params.asym_angle)
            elif self.params.asym_mask_option=='one_spider':
                self.params.asym_mask = mask_asym_baseline(self.params.asym_width,
                                                           self.params.asym_angle)
            elif self.params.asym_mask_option=='two_spiders':
                self.params.asym_mask = mask_asym_two(self.params.asym_width,
                                                      self.params.asym_angle)
            elif self.params.asym_mask_option=='two_lyot':
                self.params.asym_mask = mask_asym_two_lyot(self.params.asym_width)
            else:
                self.logger.warn('Mask option not known. Default to single bar')
                self.params.asym_mask = mask_asym_baseline(self.params.asym_width,
                                                           self.params.asym_angle)

def loadConfiguration(filename):
        '''
        Load the configuration file and return a configuration object.
        '''
        cfg_obj = Parameters(filename)

        cfg_obj.readfile()

        cfg_obj.check_parameters()
        cfg_obj.compute_parameters()

        return cfg_obj
