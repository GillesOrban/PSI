from types import SimpleNamespace
import os

# _tmp_dir = '/Users/orban/Projects/METIS/4.PSI/psi_github/data/'
_tmp_dir = os.path.dirname(__file__) + '/../data/'

conf = dict(
    # number of pixels of the pupil:
    npupil=256,     # 285
    # size of the detector plane array [lam/D]:
    det_size=10,    # 14.875,

        
    # -- Instrument name --
    # Must be a class present in ``instruments.py``
    instrument='CompassSimInstrument',

    # =======
    #   Observing Modes
    # =======
    # Instrument mode
    #    'IMG'  for no coronagraph (only telescope)
    #    'CVC'  for Classical Vortex Coronagraph
    #    'RAVC' for Ring Apodized Vortex Coronagraph
    #    WIP : 3. mode = 'APP'  for Apodizing Phase Plate
    inst_mode='IMG',
    # Vortex topological charge ('CVC' and 'RAVC')
    vc_charge=2,
    # Vector or scalar vortex ('CVC' and 'RAVC')
    # TODO support vector vortex mode
    vc_vector=False,

    # Filename for the entrance (aperture) pupil
    f_aperture=_tmp_dir + 'pupil/ELT_fullM1.fits',
    # TODO Add parameters to create aperture if no file is given

    # Filename for the Lyot stop ('CVC' or 'RAVC')
    # '''
    #     updates of pupil stops:
    #     CVC L-band: ls_CVC_L_285_dRext=0.0209_dRint=0.09_dRspi=0.0245.fits
    #     CVC N2-band: ls_CVC_N2_119_dRext=0.0268_dRint=0.09_dRspi=0.0357.fits
    #     RAVC L-band 20/05: ls_RAVC_L_285_dRext=0.0477_dRint=0.02_dRspi=0.0249.fits
    #     RAVC L-band 22/06: ls_RAVC_L_285_dRext=0.0477_dRint=0.02_dRspi=0.0249.fits

    #     f_lyot_stop = _tmp_dir + 'pupil/ls_CVC_L_285_dRext=0.0291_dRint=0.08_dRspi=0.0317.fits', # lyot stop file
    #     f_lyot_stop = _tmp_dir + 'pupil/ls_CVC_L_285_dRext=0.0209_dRint=0.09_dRspi=0.0245.fits',
    #     CVC LS L-band:
    #     f_lyot_stop = _tmp_dir+'pupil/ls_CVC_L_285_dRext=0.0209_dRint=0.09_dRspi=0.0245.fits',
    # '''
    f_lyot_stop=_tmp_dir+'pupil/ls_CVC_N2_119_dRext=0.0268_dRint=0.09_dRspi=0.0357.fits',
    # RAVC LS L-band:
    # f_lyot_stop = _tmp_dir + 'pupil/ls_RAVC_L_285_dRext=0.0477_dRint=0.04_dRspi=0.0249.fits',

    # Filename for the entrance apodization ('RAVC' and 'APP'?)
    f_apodizer=_tmp_dir + 'pupil/apo_ring_r=0.5190_t=0.7909.fits',

    # ======
    #    Photometry
    # ======
    # Adding noise:
    #   0: no noise, 
    #   1: photon noise only, 
    #   2: photon noise + background noise
    noise=2,

    # Wavelength band -- defined in photometry_definition.py
    #   Baseline for METIS: 'METIS_L' and 'METIS_N2'
    #   Alternatively if 'band' is not define, the following parameter should be passed
    #   'wavelength', 'flux_zpt', 'flux_bckg'
    band='METIS_L',

    # star magnitude at selected band
    mag=6,
    # science detector integration time [s]
    dit=0.1,

    # ======
    #  AO parameters
    # ======
    # framerate of the AO loop [Hz]
    ao_framerate=1000,
    # Decimation of the WFS telemetry use by PSI, 
    #   e.g. if =10, we use 1 every 10 WF frame
    ao_frame_decimation=10,

    # =========
    #  PSI
    # =========
    # [Hz] framerate of the psi correction
    psi_framerate=1,
    # number of iterations.
    psi_nb_iter=60,

    # How is the PSI estimate process before correction:
    #   'all'     : no projection or filtering
    #   'zern'    : projection on theoretical Zernike mode (circ ap.) and modal control
    #   'dh'    : disk harmonics
    # TODO include support to the Gendrinou basis
    psi_correction_mode='zern',
    # number of modes (if not 'all')
    psi_nb_modes=100,
    # (if modal) index of first mode. with Zernike, 4 means no piston and tip/tilt
    psi_start_mode_idx=4,

    # [nm rms] value above which the psi_correction will be skipped.
    #   set to None if no skip limit
    psi_skip_limit=None,

    # Focal plane filtering sigma (Gaussian blurring)
    #   and radius [lambda / D]
    psi_filt_sigma=0.05,
    psi_filt_radius=10,

    # PSI scaling if do not want to use 'auto scaling'
    #   default is None, otherwise expected NCPA in [nm]
    ncpa_expected_rms=None,     # 100, # 50, #250,        

    # Control gain
    gain_I = 0.2,
    gain_P = 0.1,

    # ============
    #   NCPA
    #       Only in simulation (CompassSimInstrument and HcipySimInstrument)
    # ============
    ncpa_dynamic=False,
    ncpa_sampling=100,             # [s] Dyn NCPA sampling
    ncpa_scaling=1,               # scaling factor, if want to increase level

    ncpa_folder=('/Users/orban/Projects/METIS/4.PSI/'
                 'legacy_TestArea/NCPA_Tibor/'),
    ncpa_prefix="DIFF_rep_1_field_",  # NB assumes units are in mm

    # =============
    #   Residual turbulence
    #       Only in simulation with CompassSimInstrument (offline)
    turb_folder=('/Users/orban/Projects/METIS/4.PSI/legacy_TestArea/'
                 'COMPASSPhaseScreens/ThirdAttempt_Processed/'),
    turb_prefix_rp='Residual_phase_screen_',      # NB: assumes units are in µm
    turb_prefix_wf='Reconstructed_wavefront_',    # NB: assumes units are in µm
    turb_suffix='ms_256.fits',


    # =============
    #   Water vapour seeing
    # Include water vapour seeing (bool)
    wv=False,
    wv_folder=('/Users/orban/Projects/METIS/4.PSI/legacy_TestArea/'
               'WaterVapour/phases/'),
    # WV filename (cube of phase screen)
    #   different for METIS_L and METIS_N2
    #   NB assume units are in meters
    wv_cubename=('cube_Cbasic_20210504_600s_100ms_0piston_meters_'
                 'scao_only_285_WVNonly_qacits.fits'),
    # wv_cubename = ('cube_Cbasic_20210504_600s_100ms_0piston_meters_'
    #                'scao_only_285_WVLonly_qacits.fits'),
    
    # [ms] sampling of the cube
    wv_sampling=100,
    # Scale factor, if want to change the level of WV
    wv_scaling=1,
    # =============
    # Saving results
    save_loop_statistics=True,
    save_phase_screens=False,
    save_basedir='/Users/orban/Projects/METIS/4.PSI/psi_results/',

    check_psi_convergence=False,
)

# sort alphabetically
conf = {k: v for k, v in sorted(conf.items())}
conf = SimpleNamespace(**conf)
# from attrdict import AttrDict
# conf = AttrDict(conf)
# return conf
