
from psi.configParser import loadConfiguration
from psi.instruments import CompassSimInstrument, HcipySimInstrument
import psi.psi_utils as psi_utils
from psi.deepSensor import DeepSensor

import hcipy
from astropy.visualization import imshow_norm, LogStretch, MinMaxInterval
import matplotlib.pyplot as plt
import numpy as np
import psi.deep_wfs.utils.read_data as rt

# db, attrs = rt.read_h5('/mnt/disk12tb/METIS/PSI/datasets/ds_IMG_300nm_Nband.h5')

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
# deep_sensor.buildModel('CVC_300nm_Nband_5lD_1e3')


# gen = deep_sensor.generator
# gen.setup(deep_sensor.inst, deep_sensor.C2M)

# gen.genData('toto', store_data=False,
#             phase_fname='/mnt/disk12tb/METIS/PSI/WV_screens/cube_285_300nm.fits')
#deep_sensor.next()
#deep_sensor.show()


'''
# show training loss
from psi.helperFunctions import read_config_from_text_file
dd = read_config_from_text_file('/mnt/disk12tb/METIS/PSI/models/metrics.json')

plt.figure()
plt.plot(dd.train_loss, label='training')
plt.plot(dd.val_loss, label='validation')

'''