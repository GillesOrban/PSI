
from psi.configParser import loadConfiguration
from psi.instruments import CompassSimInstrument, HcipySimInstrument
import psi.psi_utils as psi_utils

import hcipy
from astropy.visualization import imshow_norm, LogStretch, MinMaxInterval
import matplotlib.pyplot as plt
import numpy as np


config_file = 'config/config_hcipysim.py'
# config_file = 'config/config_metis_compass.py'
cfg = loadConfiguration(config_file)

inst = HcipySimInstrument(cfg.params)
inst.build_optical_model()

dt = 0.2
wfs = inst.grabWfsTelemetry(dt)
sci = inst.grabScienceImages(dt)



# # check of time buffer
# nbOfPastSeconds=0.6
# self = inst
# st = self._current_time_ms * 1e-3 
# # WFS telemetry buffer
# start_wfs = self._buffer_time_index.index(st - nbOfPastSeconds + self.sampling_time)
# end_wfs = self._buffer_time_index.index(st)

# self._start_time_wfs = self._buffer_time_index[start_wfs]
# self._end_time_wfs = self._buffer_time_index[end_wfs]






# # all sciences images
# st = self._current_time_ms * 1e-3
# start_idx = self._buffer_time_index.index(np.round(st - nbOfPastSeconds, self._decimals) ) #+ self.sampling_time)
# self._start_time_sci_buffer = self._buffer_time_index[start_idx]
# self._end_time_sci_buffer = st

# self._start_time_last_sci_dit = np.copy(self._start_time_sci_buffer)
# self._end_time_last_sci_dit = np.copy(self._start_time_sci_buffer - self.sampling_time)

# # individual DITs
# self._start_time_last_sci_dit = np.copy(self._end_time_last_sci_dit + self.sampling_time) 
# start_sci = self._buffer_time_index.index(np.round(self._start_time_last_sci_dit, self._decimals))# + self.sampling_time)
# end_sci = self._buffer_time_index.index(np.round(self._start_time_last_sci_dit +
#                                                  self.sci_exptime - self.sampling_time, self._decimals))
# self._end_time_last_sci_dit = np.copy(self._start_time_last_sci_dit + self.sci_exptime - self.sampling_time)
