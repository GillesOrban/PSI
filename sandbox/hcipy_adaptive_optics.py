from hcipy import *
import hcipy
import numpy as np
from psi.configParser import loadConfiguration
from psi.instruments import CompassSimInstrument
import psi.psi_utils as psi_utils


config_file = 'config/config_metis_compass.py'
cfg = loadConfiguration(config_file)
inst = CompassSimInstrument(cfg.params)
inst.build_optical_model()

r0=0.15
L0 = 25
wind_velocity = 8
lag = 3
delta_t=1e-3
wvl = 1 # normalized

# pupil_grid = inst.pupilGrid
diameter=40
pupil_grid = hcipy.make_pupil_grid(256, diameter=diameter)
aperture = hcipy.make_elt_aperture(normalized=False)(pupil_grid)
# Define some AO behaviour
# ao_modes = make_gaussian_influence_functions(pupil_grid, ao_actuators, 1.0 / ao_actuators)	# Create an object containing all the available DM pistons, 1.0 to
# ao_modes = ModeBasis([mode * aperture for mode in ao_modes])
# transformation_matrix = ao_modes.transformation_matrix
# reconstruction_matrix = inverse_tikhonov(transformation_matrix, reconstruction_normalisation)
nmodes = 500
zernike_modes = hcipy.make_zernike_basis(nmodes + 1, diameter, pupil_grid, radial_cutoff=False)
# mask = inst.aperture 
# mask[inst.aperture >=0.7] = 1
# mask[inst.aperture<0.7] = 0
mask = aperture
zernike_modes = psi_utils.reorthonormalize(zernike_modes, mask)

ao_modes = ModeBasis([mode * aperture for mode in zernike_modes]) # could also do some normalization

# Instantiate an atmosphere class. The idea is then to set the electric field as to that from the COMPASS residuals
layer = ModalAdaptiveOpticsLayer(InfiniteAtmosphericLayer(pupil_grid,
                                                            Cn_squared_from_fried_parameter(r0, wvl),
                                                            L0,
                                                            wind_velocity,
                                                            use_interpolation=True), 
                                   ao_modes, lag)
# atmosphere = MultiLayerAtmosphere(layers)
# atmosphere.evolve_until(t + it * t_end)

# Residual phase
time_steps = np.arange(0, 1e-2, 1e-3)
for i in time_steps:
    layer.evolve_until(i)
phase_screen_phase = layer.phase_for(wvl) # in radian
# Telemetry
coeffs = layer.transformation_matrix_inverse.dot(phase_screen_phase)
telemetry = ao_modes.linear_combination(coeffs)   


plt.figure()
hcipy.imshow_field(phase_screen_phase * aperture)


wf = Wavefront(aperture)
wf.total_power = num_photons
wf_post_ao = atmosphere(wf)

wfs_measurement = reconstruction_matrix.dot(np.angle(wf_post_ao.electric_field / wf_post_ao.electric_field.mean()) * aperture)  # * aperture is different between METIS and ERIS
wfs_measurement_noisy = wfs_measurement * (1 + np.random.randn(len(wfs_measurement)) * wfs_noise)