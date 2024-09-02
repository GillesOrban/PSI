
from psi.configParser import loadConfiguration
from psi.instruments import CompassSimInstrument, HcipySimInstrument
import psi.psi_utils as psi_utils
from psi.deepSensor import DeepSensor

import hcipy
from astropy.visualization import imshow_norm, LogStretch, SqrtStretch,\
    MinMaxInterval
import matplotlib.pyplot as plt
import numpy as np
import psi.deep_wfs.utils.read_data as rt
import astropy.io.fits as fits
import glob, os

check_dataset = False
check_datasets_attrs = False
check_wavefront_projection = False
check_scao_modal_basis = True
test_model = False
test_model_with_dataset = False

if check_dataset:
    # Check dataset
    datadir = '/mnt/disk12tb/METIS/PSI/datasets/'
    # f_dataset = 'ds_METIS_N2_CVC_mag=-2_bw=0.0_mask=two_lyot_15%_Z20_s1e+04_r1.h5'
    f_dataset = 'ds_METIS_L_SPP_mag=6_bw=0.0_mask=AP2_Z100_s1e+04_nds.h5'
    db, attrs = rt.read_h5(datadir + f_dataset)

    print('Input WV phase screen cube : {0}'.format(attrs['phasescreen_fname']))
    if 'noisy' in attrs:
        print('Is dataset noisy? : {0}'.format(attrs['noisy']))
    else:
        print('Noise configuration is : {0}'.format(attrs['noise']))

    pupil_stop = db['asymmetric_stop'].reshape((256, 256))
    psfs = db['psfs_1']
    zern = db['zernike_coefficients']

    plt.figure()
    imshow_norm(psfs[0], stretch=SqrtStretch())
    plt.colorbar()

if check_datasets_attrs:
    datadir = '/mnt/disk12tb/METIS/PSI/datasets/'
    fnames = glob.glob(datadir + 'ds_METIS*.h5')
    fnames.sort(key=os.path.getmtime)
    print(fnames)

    for i in range(len(fnames)):
        print('\n-- file {0}'.format(fnames[i]))
        _, attrs = rt.read_h5(fnames[i], attrs_only=True)
        try:
            print('Input WV phase screen cube : {0}'.format(attrs['phasescreen_fname']))
        except:
            print('no phasescreen name')
        if 'noisy' in attrs:
            print('Is dataset noisy? : {0}'.format(attrs['noisy']))
        else:
            print('Noise configuration is : {0}'.format(attrs['noise']))


if check_wavefront_projection:
    # check wavefront cube and projection
    config_file='config/config_deep_learning.py'
    ncpa_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/NCPA_Tibor/'
    turb_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/COMPASSPhaseScreens/ThirdAttempt_Processed/'
    wv_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/WaterVapour/phases/'

    datadir = '/mnt/disk12tb/METIS/PSI/WV_screens/'
    fname_phase_dataset = 'cube_285_300nm_1e5.fits'
    cube_phase_data = fits.getdata(datadir + fname_phase_dataset)


    fname_phase_wv = 'cube_Cbasic_20210504_600s_100ms_0piston_meters_scao_only_285_WVNonly_qacits.fits'
    cube_phase_wv = fits.getdata(wv_folder +  fname_phase_wv)

    save_basedir='/home/gorban/'
    deep_sensor = DeepSensor(config_file)
    deep_sensor.cfg.params.ncpa_folder = ncpa_folder
    deep_sensor.cfg.params.turb_folder = turb_folder
    deep_sensor.cfg.params.wv_folder = wv_folder
    deep_sensor.cfg.params.save_basedir = save_basedir
    deep_sensor.setup()
    gen = deep_sensor.generator
    inst = deep_sensor.inst
    gen.setup(inst, deep_sensor.C2M, deep_sensor.cfg.params)




    inst.include_residual_turbulence = False
    inst.include_water_vapour = False
    inst.ncpa_dynamic = False
    inst.phase_wv *= 0
    inst.phase_ncpa *= 0
    gen.phase_cube = cube_phase_data

    idx = 0
    phase_screen = gen._get_phase_screen(idx)
    inst.phase_ncpa = phase_screen
    nbOfSeconds = 0.1  # 0.1 seconds is the shortest possible with CompassSim
    science_images_buffer = inst.grabScienceImages(nbOfSeconds)
    science_image = science_images_buffer.mean(0)

    modes = gen._C2M.dot(phase_screen) #* gen.config['wavelength']/ (2 * np.pi)  # nm

    wf_modes = deep_sensor.M2C.dot(modes).T.reshape((256, 256))


    # modal stddev depends on aperture definition and the C2M
    # -> with aper not completely Noll but well behaved.
    nmodes=100    
    nmode_shift=1
    reortho=True

    # -- ELT aperture
    # aper = inst.aperture

    # -- Annular aperture that contains the ELT aperture
    grid_diam = inst.pupilGrid.delta[0] * inst.pupilGrid.dims[0]
    aper = hcipy.aperture.make_circular_aperture(0.98 * grid_diam)(inst.pupilGrid)
    aper -= hcipy.aperture.make_circular_aperture(0.25 * grid_diam)(inst.pupilGrid)
    tmpGrid = inst.pupilGrid.copy().scale(1/grid_diam)

    M2C, C2M = psi_utils.makeModalBasis(inst.pupilGrid,
                                        nmodes,
                                        nmode_shift,
                                        reortho=reortho,
                                        aperture=aper,
                                        basis_name='zern')
    gen._C2M = C2M

    nentries=1000
    print('Modal projection on phase cube of dataset {0}'.format(fname_phase_dataset))
    gen.phase_cube = cube_phase_data
    modes_cube_data = np.zeros((nentries, nmodes))
    for i in range(nentries):
        phase_screen = gen._get_phase_screen(i)
        modes_cube_data[i,:] = gen._C2M.dot(phase_screen)
    std_modes_data = np.std(modes_cube_data, axis=0)

    print('Modal projection on WV phase cube {0}'.format(fname_phase_wv))
    gen.phase_cube = cube_phase_wv

    modes_cube_wv = np.zeros((nentries, nmodes))
    for i in range(nentries):
        phase_screen = gen._get_phase_screen(i)
        modes_cube_wv[i,:] = gen._C2M.dot(phase_screen)
    std_modes_wv = np.std(modes_cube_wv, axis=0)

    plt.figure()
    plt.plot(np.arange(nmode_shift+1, nmodes + nmode_shift + 1),
             std_modes_data, label='phase data')
    plt.plot(np.arange(nmode_shift+1, nmodes + nmode_shift + 1),
             std_modes_wv, label='phase WV')
    plt.semilogy()
    plt.legend()

    mean_rms_wfe_data = np.sqrt(np.mean(np.sum(modes_cube_data**2, axis=1))) * gen.config['wavelength']/ (2 * np.pi)
    mean_rms_wfe_wv = np.sqrt(np.mean(np.sum(modes_cube_wv**2, axis=1))) * gen.config['wavelength']/ (2 * np.pi)

    print('Mean rms WFE on data = {0:.1f}nm'.format(mean_rms_wfe_data))
    print('Mean rms WFE on WV = {0:.1f}nm'.format(mean_rms_wfe_wv))

    # # aperture of 'initModalBasis''
    # # TODO:
    # #   - should I remove the asym_mask in the case of VC ? and in the case of IMG?
    # grid_diam = inst.pupilGrid.delta[0] * inst.pupilGrid.dims[0]
    # aper = hcipy.aperture.make_circular_aperture(0.98 * grid_diam)(inst.pupilGrid)
    # aper -= hcipy.aperture.make_circular_aperture(0.25 * grid_diam)(inst.pupilGrid)
    # tmpGrid = inst.pupilGrid.copy().scale(1/grid_diam)
    # aper *= inst._asym_mask(tmpGrid)


if check_scao_modal_basis:
    # check wavefront cube and projection
    config_file='config/config_deep_learning.py'
    ncpa_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/NCPA_Tibor/'
    turb_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/COMPASSPhaseScreens/ThirdAttempt_Processed/'
    wv_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/WaterVapour/phases/'

    datadir = '/mnt/disk12tb/METIS/PSI/WV_screens/'
    fname_phase_dataset = 'cube_285_300nm_1e5.fits'
    cube_phase_data = fits.getdata(datadir + fname_phase_dataset)


    fname_phase_wv = 'cube_Cbasic_20210504_600s_100ms_0piston_meters_scao_only_285_WVNonly_qacits.fits'
    cube_phase_wv = fits.getdata(wv_folder +  fname_phase_wv)

    save_basedir='/home/gorban/'
    deep_sensor = DeepSensor(config_file)
    deep_sensor.cfg.params.ncpa_folder = ncpa_folder
    deep_sensor.cfg.params.turb_folder = turb_folder
    deep_sensor.cfg.params.wv_folder = wv_folder
    deep_sensor.cfg.params.save_basedir = save_basedir
    deep_sensor.setup()
    gen = deep_sensor.generator
    inst = deep_sensor.inst
    gen.setup(inst, deep_sensor.C2M, deep_sensor.cfg.params)

    # scao_modes_raw = fits.getdata(datadir + '../modal_basis/20240708_Dfull_102modes_256_derot.fits')
    # # renormalization of the modes
    # scao_modes_raw *= 1e6
    # scao_basis = hcipy.ModeBasis(np.reshape(scao_modes_raw, (102, 256*256)).T, 
    #                                          deep_sensor.inst.pupilGrid)

    # nmode_shift=2 # skip tip and tilt; the scao_basis does not include piston unlike in HCIPy
    # M2C = scao_basis.transformation_matrix[:, nmode_shift:]
    # C2M =hcipy.inverse_tikhonov(scao_basis.transformation_matrix,
    #                             1e-3)[nmode_shift:,:]

    # extract modes from M2C 
    modes = np.reshape(deep_sensor.M2C, (256, 256, 20))
    



if test_model:
    #model_tag = 'METIS_N2_CVC_mag=-2_bw=0.0_mask=two_lyot_20%_Z20_s1e+04_r1'
    #model_tag = 'METIS_N2_CVC_mag=-2_bw=0.0_mask=one_spider_20%_Z20_s1e+04_nds'
    # model_tag = 'METIS_N2_CVC_mag=-2_bw=0.0_mask=two_lyot_20%_Z20_s1e+04_nds_v1_t0'
    # model_tag = 'METIS_N2_CVC_mag=0_bw=0.0_mask=two_lyot_20%_Z20_s1e+04_nds'
    model_tag = 'METIS_N2_CVC_mag=-4_bw=0.0_mask=two_lyot_20%_Z20_s1e+04_nds'

    dir_model = '/mnt/disk12tb/METIS/PSI/models/'
    model_path = dir_model + model_tag + '/'
    config_file='config/config_deep_learning.py'
    ncpa_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/NCPA_Tibor/'
    turb_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/COMPASSPhaseScreens/ThirdAttempt_Processed/'
    
    wv_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/WaterVapour/phases/'
    wv_cubename=('cube_Cbasic_20210504_600s_100ms_0piston_meters_'
                 'scao_only_285_WVNonly_qacits.fits')

    # wv_folder= '/mnt/disk12tb/METIS/PSI/WV_screens/'
    # wv_cubename = ('cube_285_300nm_1e5.fits')

    save_basedir='/home/gorban/'
    deep_sensor = DeepSensor(config_file)
    deep_sensor.cfg.params.ncpa_folder = ncpa_folder
    deep_sensor.cfg.params.turb_folder = turb_folder
    deep_sensor.cfg.params.wv_folder = wv_folder
    deep_sensor.cfg.params.wv_cubename = wv_cubename
    deep_sensor.cfg.params.save_basedir = save_basedir
    deep_sensor.setup()
    gen = deep_sensor.generator
    inst = deep_sensor.inst

    # inst.include_residual_turbulence = False
    # inst.include_water_vapour = True
    # inst.ncpa_dynamic = False
    # # inst.phase_wv *= 0
    # inst.phase_ncpa *= 0


    # evaluator = deep_sensor.evaluator

    # evaluator.setup(model_data_path=model_path)
    deep_sensor.init_evaluator(model_fname=model_path)

    deep_sensor.evaluateSensorEstimate()

    nb_modes=20
    modal_gains=np.linspace(0.5, 1, num=nb_modes)[::-1]

if test_model_with_dataset:
    model_tag = 'METIS_N2_CVC_mag=-2_bw=0.0_mask=two_lyot_20%_Z20_s1e+04_r1'
    f_dataset = 'ds_METIS_N2_CVC_mag=-2_bw=0.0_mask=two_lyot_20%_Z20_s1e+04_r1.h5'
    ds_datadir = '/mnt/disk12tb/METIS/PSI/datasets/'


    dir_model = '/mnt/disk12tb/METIS/PSI/models/'
    model_path = dir_model + model_tag + '/'
    config_file='config/config_deep_learning.py'
    ncpa_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/NCPA_Tibor/'
    turb_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/COMPASSPhaseScreens/ThirdAttempt_Processed/'
    wv_folder='/mnt/disk12tb/METIS/PSI/legacy/TestArea/WaterVapour/phases/'

    save_basedir='/home/gorban/'
    deep_sensor = DeepSensor(config_file)
    deep_sensor.cfg.params.ncpa_folder = ncpa_folder
    deep_sensor.cfg.params.turb_folder = turb_folder
    deep_sensor.cfg.params.wv_folder = wv_folder
    deep_sensor.cfg.params.save_basedir = save_basedir
    deep_sensor.setup()
    deep_sensor.init_evaluator(model_fname=model_path)
    evaluator = deep_sensor.evaluator

    db, attrs = rt.read_h5(ds_datadir + f_dataset)
    psfs = db['psfs_1']

    nzern=20
    n_tests=1000
    modes_truth = np.zeros((n_tests, nzern))
    modes_meas = np.zeros((n_tests, nzern))
    noise = 2
    num_photons = deep_sensor.getFluxInFocalPlane() / 10000 #self.inst.num_photons
    bckg_level = deep_sensor.inst.bckg_level
    for i in range(n_tests):
        modes_truth[i] = db['zernike_coefficients'][i]
        modes_meas[i] = evaluator.infer(psfs[i][np.newaxis, :, :],
                                        noise=noise,
                                        signal=num_photons,
                                        bckg=bckg_level).squeeze()

    err = np.sqrt(np.mean((modes_meas - modes_truth)**2, axis=0))
    std_truth = np.std(modes_truth, axis=0)

    plt.figure()
    plt.plot(err, label='res. error')
    plt.plot(std_truth, label='input')
    plt.legend()
    plt.semilogy()
    print('Total error {0:.1f}nm'.format(np.sqrt(np.sum(err**2))))


modes_truth = np.array(deep_sensor._log_modes_truth)
modes_meas = np.array(deep_sensor._log_modes_meas)

std_truth=np.std(modes_truth, axis=0)
std_meas=np.std(modes_meas, axis=0)