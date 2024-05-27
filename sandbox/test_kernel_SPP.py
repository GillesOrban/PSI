from psi.kernelSensor import KernelSensor
import numpy as np
config_file='config/config_metis_compass_kernel_SPP.py'
kernel = KernelSensor(config_file)

kernel.setup()

nb_modes=100
kernel._initModalBases(nb_modes)

# gain_I=0.9 #0.45
# gain_P=0 #0.45
modal_gains=np.linspace(0.5, 1, num=nb_modes)[::-1]

kernel.evaluateSensorEstimate()

# kernel.loop(leak=1, modal_gains=modal_gains)
kernel.next()