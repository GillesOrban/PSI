
from psi.configParser import loadConfiguration
from psi.instruments import CompassSimInstrument, HcipySimInstrument
import psi.psi_utils as psi_utils
from psi.deepSensor import DeepSensor

import hcipy
from astropy.visualization import imshow_norm, LogStretch, MinMaxInterval
import matplotlib.pyplot as plt
import numpy as np


config_file='config/config_deep_learning.py'

deep_sensor = DeepSensor(config_file)
deep_sensor.setup()

