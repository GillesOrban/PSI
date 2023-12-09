


# from heeps.util.download_from_gdrive import extract_zip
import astropy.units as u
import os
import numpy as np
from types import SimpleNamespace
# import proper
# proper.print_it = False


# def read_config(verbose=False, **update_conf):
_tmp_dir = 'data/'

conf = dict(
    # number of pixels of the pupil
    npupil = 256 , 
    # [lam/D] radial size of the detector plane array                      
    det_size = 15 ,                   


    # --- Which type of instrument to use --
    # Must be a class present in ``instruments.py``
    instrument = 'HcipySimInstrument', #'HcipySimInstrument',
    # [HcipySimInstrument] only - Pupil type:'ERIS' or 'ELT', or 'CIRC'
    # NB: for ELT, if f_aperture is given, will use it, otherwise use HCIPy to define the ELT aperture
    pupil = 'ELT',  

    # =======
    #   Observing Modes
    # =======
    # Instrument mode
    #    'IMG'  for no coronagraph (only telescope)
    #    'CVC'  for Classical Vortex Coronagraph
    #    'RAVC' for Ring Apodized Vortex Coronagraph
    #    WIP : 3. mode = 'APP'  for Apodizing Phase Plate
    inst_mode = 'IMG',                  
    # Vortex topological charge ('CVC' and 'RAVC')
    vc_charge = 2,              
    # Vector or scalar vortex ('CVC' and 'RAVC')       
    vc_vector = False,     

    # Filename for the entrance (aperture) pupil            
    f_aperture=_tmp_dir + 'pupil/ELT_fullM1.fits',


    # ======
    #    Photometry
    # ======
    # Adding noise:
    #   0: no noise, 
    #   1: photon noise only, 
    #   2: photon noise + background noise
    noise = 0  ,                       

    # Wavelength band -- defined in photometry_definition.py
    #   Baseline for METIS: 'METIS_L' and 'METIS_N2'
    #   Alternatively if 'band' is not define, the following parameter should be passed
    #   'wavelength', 'flux_zpt', 'flux_bckg'
    band='METIS_N2',

    # star magnitude at selected band
    mag=-1.5,
    # Polychromatic bandwidth
    bandwidth=0.0,
    # science detector integration time [s]
    dit=0.1,

    # ======
    #  AO parameters
    # ======
    # framerate of the AO loop [Hz]
    ao_framerate = 1000 ,       
    # Decimation of the WFS telemetry use by PSI, 
    #   e.g. if =10, we use 1 every 10 WF frame
    ao_frame_decimation = 10,    

    # ========
    # Generic FP Sensor parameters
    # ========
    # Number of modes sensed and corrected.
    nb_modes = 100,      # (generic `psi_nb_modes`
    # Number of iteration. Total duration is nb_iter / framerate
    nb_iter = 600,      # (generic `nb_iter`
     # [Hz] framerate of the sensing & correction
    framerate = 10,     # (generic `psi_framerate`

    # modal basis: zern, dh, gendrinou
    modal_basis = 'gendrinou',

    # Control gains
    gain_I=0.45, #0.2 for IMG # Integrator gain
    gain_P=0.45, #0.1 for IMG # Proportional gain

    # Saving results
    save_loop_statistics=False,
    save_phase_screens=False,
    save_basedir='/home/gorban/', #'/Users/orban/Projects/METIS/4.PSI/psi_results/',
    save_dirname=None,  # TODO: explain difference with save_basedir


    # =========
    #  Kernel
    # =========
    asym_stop=True,
    asym_angle=180,                   # [optional]
    asym_width=0.15,                  # [optional]
    # Asymmetric mask configuration. 
    # Currently implemented: 'one_spider', 'two_spiders', 'two_lyot'
    asym_mask_option= 'two_spiders',
    asym_model_fname=None,            # [optional]
    # nb of steps along the pupil diameter
    asym_nsteps=33,
    # transmission min
    asym_tmin=0.5,

    # # =========
    # #  PSI
    # # =========
    # TODO make 'framerate' sensor agnostics -> renaming
    # psi_framerate = 10,           # [Hz] framerate of the psi correction
    # psi_nb_iter = 60,            # number of iterations.

    # # How is the PSI estimate process before correction:
    # #   1. all     : no projection or filtering
    # #   2. zern    : projection on theoretical Zernike mode (circ ap.) and modal control
    # #   3. dh    : disk harmonics

    # psi_correction_mode = 'zern',
    # psi_nb_modes = 100,           # (if modal) nb of modes
    # psi_start_mode_idx = 4,        # (if modal) index of first mode. with Zernike, 4 means no piston and tip/tilt

    # psi_skip_limit = None,         # [nm rms] value above which the psi_correction will be skipped.
    #                               # set to None if no skip limit

    # # Focal plane filtering
    # psi_filt_sigma = 0.05,
    # psi_filt_radius = 10,          # [lbda/D]

    # # PSI scaling --- because of unknown scaling factor of NCPA
    # ncpa_expected_rms = 80, #250,        # expected NCPA in [nm]

    # check_psi_convergence = False,

    # ============
    #   NCPA
    #       Only in simulation (CompassSimInstrument and HcipySimInstrument)
    # ============
    ncpa_dynamic =  False ,
    ncpa_sampling = 100,             # [s] Dyn NCPA sampling
    ncpa_scaling = 0.,               # scaling factor, if want to increase level
    ncpa_coefficients = [0],
    # =============
    #   Residual turbulence
    residual_turbulence = False,  # TODO to implement in the configParser and in the inst
    # r0 = 0.15,
    # L0 = 25,
    # nmodes = 500,
    ##...



)
    # sort alphabetically
conf = {k: v for k, v in sorted(conf.items())}
conf = SimpleNamespace(**conf)
# from attrdict import AttrDict
# conf = AttrDict(conf)
    # return conf
