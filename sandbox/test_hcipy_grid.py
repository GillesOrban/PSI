import hcipy
import numpy as np
import matplotlib.pyplot as plt

pupil_size = 256
diameter=1


pupil_grid = hcipy.make_pupil_grid(pupil_size, diameter=diameter)
focal_grid = hcipy.make_focal_grid(4, 16,
                                   pupil_diameter=diameter,
                                   reference_wavelength=1,
                                   focal_length=1)
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

norm=False
if diameter==1:
    norm=True
aperture = hcipy.make_elt_aperture(normalized=norm)(pupil_grid)

img = prop(hcipy.Wavefront(aperture * np.exp(1j * 0))).power


hcipy.imshow_field(img)

