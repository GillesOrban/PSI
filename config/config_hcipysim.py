


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

    npupil = 256, #285,                        # number of pixels of the pupil
    det_size = 15, #14.875,                      # [lam/D] radial size of the detector plane array

    # det_res should be None by default and computed based on the band_specs provdied below
    # this in order to have the correct sampling wrt to the background noise.
    det_res = 4,                       # [px/ (lbda/D)] number of pixels per resolution element
                                        #~4 px in L-band; 9.3 in N-band
    # --- Which type of instrument to use --
    # Must be a class present in ``instruments.py``
    instrument = 'HcipySimInstrument',
    pupil = 'CIRC',   # 'ERIS' or 'ELT', or 'CIRC'

    # =======
    #   Observing Modes
    # =======
    #    0. mode = 'ELT'  for no coronagraph (only telescope)
    #    1. mode = 'CVC'  for Classical Vortex Coronagraph
    #    (2. mode = 'RAVC' for Ring Apodized Vortex Coronagraph)
    #    (3. mode = 'APP'  for Apodizing Phase Plate)
    # TODO replace mode name as follows:
    #   ELT -> IMG_LM or IMG_N
    #   CVC -> CVC_LM or CVC_N
    #   RAVC -> RAVC_LM
    inst_mode = 'ELT',                  # HCI instrument mode
    vc_charge = 2,                      # (CVC and RAVC only) vortex topological charge
    vc_vector = False,                  # (CVC and RAVC only) simulate a vector vortex instead of a scalar one


    # ======
    #    Photometry
    # ======
    noise = 2  ,                        # 0: no noise, 1: photon noise only, 2: photon noise + background noise
    mag = 0,                            # star magnitude at selected band

    # --- the 3 following parameters should be replaced by the 'band_specs provided below'
    ## L-filter
    wavelength = 3.81e-6,               # [m] wavelength
    flux_zpt = 8.999e+10,               # [e-/s] zeropoint HCI-L long, mag 0 (Jan 21, 2020)
    flux_bckg = 8.878e+4,               # [e-/s/pix]

    ## N2-filter
    # wavelength = 11.33e-6, #3.81e-6   ,         # [m] wavelength
    # flux_zpt = 3.695e10, #8.999e+10,            # [e-/s] zeropoint HCI-L long, mag 0 (Jan 21, 2020)
    # flux_bckg = 1.122e8, #8.878e+4,             # [e-/s/pix]

    bandwidth = 0.,                            # bandwidth for polychromatic simulations, 0 for monochromatic
    dit = 0.1,                          # [s] science detector integration time


    # ======
    #  AO parameters
    # ======
    ao_framerate = 1000 ,        # [Hz] framerate of the AO loop
    ao_frame_decimation = 10,    # Decimation of the WFS telemetry use by PSI, e.g. if =10, we use 1 every 10 WF frame

    # =========
    #  Kernel
    # =========
    asym_stop = True,
    asym_angle = 90,                   # [optional]
    asym_width = 0.15,                  # [optional]
    asym_model_fname = None, #toto.fits.gz',              # [optional]
    # asym_telDiam = 40,
    asym_nsteps=33, # nb of steps along the pupil diameter
    asym_tmin=0.5, # transmission min

    # # =========
    # #  PSI
    # # =========
    # TODO make 'framerate' sensor agnostics -> renaming
    psi_framerate = 10,           # [Hz] framerate of the psi correction
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

    # ============
    #   NCPA
    #       Only in simulation (CompassSimInstrument and HcipySimInstrument)
    # ============
    ncpa_dynamic =  False ,
    ncpa_sampling = 100,             # [s] Dyn NCPA sampling
    ncpa_scaling = 0.,               # scaling factor, if want to increase level

    # =============
    #   Residual turbulence
    residual_turbulence = False,  # TODO to implement in the configParser and in the inst
    # r0 = 0.15,
    # L0 = 25,
    # nmodes = 500,
    ##...

    # =============
    # Saving results
    save_loop_statistics = True,
    save_phase_screens = False,
    save_basedir = '/Users/orban/Projects/METIS/4.PSI/psi_results/',

    check_psi_convergence = False,

)
    # sort alphabetically
conf = {k: v for k, v in sorted(conf.items())}
conf = SimpleNamespace(**conf)
# from attrdict import AttrDict
# conf = AttrDict(conf)
    # return conf
