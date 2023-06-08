import hcipy
import numpy as np
import matplotlib.pyplot as plt


pupil_grid = hcipy.make_pupil_grid(256)

telescope_pupil_generator = hcipy.make_magellan_aperture(normalized=True)

telescope_pupil = telescope_pupil_generator(pupil_grid)

im = hcipy.imshow_field(telescope_pupil, cmap='gray')
plt.colorbar()
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()




wavefront = hcipy.Wavefront(telescope_pupil)

focal_grid = hcipy.make_focal_grid(q=8, num_airy=16)
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

focal_image = prop.forward(wavefront)

plt.figure()
hcipy.imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)
plt.xlabel('Focal plane distance [$\lambda/D$]')
plt.ylabel('Focal plane distance [$\lambda/D$]')
plt.colorbar()
plt.show()


plt.figure()
bandwidth = 0.2

focal_total = 0
for wlen in np.linspace(1 - bandwidth / 2., 1 + bandwidth / 2., 11):
    wavefront = hcipy.Wavefront(telescope_pupil, wlen)
    focal_total += prop(wavefront).intensity

hcipy.imshow_field(np.log10(focal_total / focal_total.max()), vmin=-5)

plt.title('Magellan PSF with a bandwidth of {:.1f} %'.format(bandwidth * 100))
plt.colorbar()
plt.xlabel('Focal plane distance [$\lambda/D$]')
plt.ylabel('Focal plane distance [$\lambda/D$]')
plt.show()


def orthonormalize(modal_basis, aperture):
    transformation_matrix = modal_basis.transformation_matrix * aperture[:, np.newaxis]
    # q, r = np.linalg.qr(transformation_matrix, mode='complete')
    
    nvalid_pixels = np.sum(aperture)
    cross_product = 1 / nvalid_pixels * np.dot(transformation_matrix.T, transformation_matrix)
    L = np.linalg.cholesky(cross_product)

    basis_orthonormalized = np.dot(transformation_matrix, np.linalg.inv(L).T)

    return hcipy.ModeBasis(basis_orthonormalized, modal_basis.grid)

idx=[5]
coeffs=[1]

zernike_modes = hcipy.make_zernike_basis(np.max(idx)+1, 1, pupil_grid, radial_cutoff=False)
coefficient_vector = np.zeros(np.max(idx)+1)
coefficient_vector[idx] = coeffs
aberration = zernike_modes.linear_combination(coefficient_vector)
# np.std(aberration_sq[aberration != 0])

# Monochromatic
wavefront = hcipy.Wavefront(telescope_pupil * np.exp(1j * aberration), 1)
focal_total_mono = prop(wavefront).intensity

# Monochromatic - short
wlen = 0.5
wavefront = hcipy.Wavefront(telescope_pupil * np.exp(1j * aberration /wlen), wlen)
focal_total_mono_short = prop(wavefront).intensity

# # Polychromatic
# bandwidth = 0.2
# focal_total = 0
# for wlen in np.linspace(1 - bandwidth / 2., 1 + bandwidth / 2., 11):
#     wavefront = hcipy.Wavefront(telescope_pupil * np.exp(1j * aberration), wlen)
#     focal_total += prop(wavefront).intensity

plt.figure()
plt.subplot(121)
hcipy.imshow_field(np.log10(focal_total_mono / focal_total_mono.max()), vmin=-5)

# plt.title('Magellan PSF with a bandwidth of {:.1f} %'.format(bandwidth * 100))
# plt.colorbar()
# plt.xlabel('Focal plane distance [$\lambda/D$]')
# plt.ylabel('Focal plane distance [$\lambda/D$]')
plt.subplot(122)
hcipy.imshow_field(np.log10(focal_total_mono_short / focal_total_mono_short.max()), vmin=-5)

plt.show()