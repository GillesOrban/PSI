import hcipy
import numpy as np
import matplotlib.pyplot as plt
import psi.psi_utils as psi_utils

test_hcipy_fourier = False
test_custom_fourier = True

# Fourier modes from HCIPy
if test_hcipy_fourier:
    pupil_size = 128
    pupil_grid = hcipy.make_pupil_grid(pupil_size)

    telescope_pupil_generator = hcipy.make_magellan_aperture(normalized=True)

    telescope_pupil = telescope_pupil_generator(pupil_grid)

    plt.figure()
    im = hcipy.imshow_field(telescope_pupil, cmap='gray')
    plt.colorbar()
    plt.xlabel('x / D')
    plt.ylabel('y / D')
    plt.show()

    wavefront = hcipy.Wavefront(telescope_pupil)

    focal_grid = hcipy.make_focal_grid(q=4, num_airy=8)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

    focal_image = prop.forward(wavefront)


    fourier_modes = hcipy.make_fourier_basis(pupil_grid, focal_grid, sort_by_energy=True)

    basis = np.zeros((fourier_modes.num_modes, pupil_size, pupil_size))
    for i in range(fourier_modes.num_modes):
        basis[i] = fourier_modes[i].shaped


# Custom Fourier modal basis
if test_custom_fourier:
    pupil_size = 128
    pupil_grid = hcipy.make_pupil_grid(pupil_size)
    aperture = hcipy.make_circular_aperture(1)(pupil_grid)

    par, fourier = psi_utils.fourier_modes_simple(pupil_grid, aperture,
                                                k=[1, 10], q=4)

    plt.figure()
    plt.imshow(fourier[73].reshape((128,128)))

    # def fourier_nm(n: int, m: int,  N: int, p:int=1 or -1,
    #                 X=None, Y=None):
    #     if X is None and Y is None:
    #         coords = (np.arange(N) - N / 2. + 0.5) / (N / 2.)
    #         X, Y = np.meshgrid(coords, coords)
        
    #     M = np.cos(2 * np.pi * (n * X + m * Y) / 2) + \
    #         p * np.sin(2 * np.pi * (n * X + m * Y) / 2)
    #     # return aotools.circle(N / 2., N) * M
    #     return M

    # k_start = 1
    # k_end = 10
    # q = 4 # number of pixels below lbda/D
    # kx_list = np.arange(k_start, k_end + 1/q, 1/q)
    # ky_list = np.array([0]*len(kx_list))
    # kxy_list = np.concatenate((kx_list[:, np.newaxis], ky_list[:, np.newaxis]), axis=1)

    # mode_basis_cube = np.zeros((kxy_list.shape[0]*2, pupil_size, pupil_size))
    # for i in range(kxy_list.shape[0]*2):
    #     p = -1
    #     if not(i % 2):
    #         p = 1
    #     mode_basis_cube[i] = aperture.shaped * fourier_nm(kx_list[i//2],
    #                                 ky_list[i//2],
    #                                 pupil_size, p)
    #                                 # X=pupil_grid.x.reshape((pupil_size, pupil_size)),
    #                                 # Y=pupil_grid.y.reshape((pupil_size,pupil_size)))
    # plt.figure()
    # plt.imshow(mode_basis_cube[10])

    # modeBasis = hcipy.ModeBasis([mode.flatten() for mode in mode_basis_cube], pupil_grid)

    # plt.figure()
    # plt.imshow(modeBasis[10].reshape((pupil_size, pupil_size)))
