from psi.configParser import loadConfiguration
from psi.instruments import CompassSimInstrument, HcipySimInstrument
import psi.psi_utils as psi_utils

import hcipy
from astropy.visualization import imshow_norm, LogStretch, MinMaxInterval
import matplotlib.pyplot as plt
import numpy as np

test_metis_compass_inst = False
test_hcipy_sim_inst = True

if test_metis_compass_inst:
    config_file = 'config/config_metis_compass.py'
    cfg = loadConfiguration(config_file)
    inst = CompassSimInstrument(cfg.params)
    inst.build_optical_model()

if test_hcipy_sim_inst:
    # config_file = 'config/config_hcipy_sim.py'
    config_file = 'config/config_metis_compass.py'
    cfg = loadConfiguration(config_file)

    inst = HcipySimInstrument(cfg.params)
    inst.build_optical_model()

    inst.include_residual_turbulence = False
    inst.bandwidth=0.0

    par, fourier = psi_utils.fourier_modes_simple(inst.pupilGrid, inst.aperture,
                                                k=[1, 10], q=4)
    idx = np.argwhere(np.array(par['m']) == 10)[0][0]
    ampl=0.1
    inst.set_ncpa(fourier[idx] * ampl)    
    # coeffs = np.zeros(inst.ao_modes.num_modes)
    # coeffs[5] = 1
    # inst._initialize_ncpa(coeffs)

    arr = inst.grabScienceImages(0.1)


    plt.figure()
    imshow_norm(arr[0], interval=MinMaxInterval(), stretch=LogStretch())

