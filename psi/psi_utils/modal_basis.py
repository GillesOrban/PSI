import numpy as np
import hcipy

def reorthonormalize(modal_basis, aperture):
    transformation_matrix = modal_basis.transformation_matrix * aperture[:, np.newaxis]
    # q, r = np.linalg.qr(transformation_matrix, mode='complete')
    
    nvalid_pixels = np.sum(aperture)
    cross_product = 1 / nvalid_pixels * np.dot(transformation_matrix.T, transformation_matrix)
    L = np.linalg.cholesky(cross_product)

    basis_orthonormalized = np.dot(transformation_matrix, np.linalg.inv(L).T)

    return hcipy.ModeBasis(basis_orthonormalized, modal_basis.grid)


def fourier_modes_simple(pupil_grid, aperture, k = [1, 10], q=4):
    '''
        Convenient method to generate Fourier modes.
        This is not intended as a complete basis (see e.g. HCIPy Fourier basis for that)
        but a convenient method to generate intuitive Fourier modes 

        Currently generate only horizontal modes

        Parameters
        ----------
        pupil grid : hcipy CartesianGrid
        aperture   : hcipy field or 2D numpy array
        k          : list
            min and max frequency 
        q           : float
            1 / frequency sampling. 
            Should correspond to the number of pixels per lbda/D in the detector plane
        
        Returns
        -------
        params      : dic
            {'m', 'n', 'p'} the x-axis and y-axis frequencies, and p = +1 or -1
        basis       : hcipy.ModeBasis

    '''
    def fourier_nm(n: int, m: int,  N: int, p:int=1 or -1,
                    X=None, Y=None):
        '''
            Normally X,Y do not need to be recalculated and could be extracted from pupil_grid:
                pupil_grid.x.reshape((pupil_size, pupil_size))
            ...

            see also
                Males and Guyon 2018, eq. 17
        '''
        if X is None and Y is None:
            coords = (np.arange(N) - N / 2. + 0.5) / (N / 2.)
            X, Y = np.meshgrid(coords, coords)
        
        M = np.cos(2 * np.pi * (n * X + m * Y) / 2) + \
            p * np.sin(2 * np.pi * (n * X + m * Y) / 2)
        # return aotools.circle(N / 2., N) * M
        return M

    kx_list = np.arange(k[0], k[1] + 1/q, 1/q)
    ky_list = np.array([0]*len(kx_list))
    kxy_list = np.concatenate((kx_list[:, np.newaxis], ky_list[:, np.newaxis]), axis=1)

    pupil_size = pupil_grid.dims[0]
    mode_basis_cube = np.zeros((kxy_list.shape[0]*2, pupil_size, pupil_size))

    params = {'m':[], 'n':[], 'p':[]}

    for i in range(kxy_list.shape[0]*2):
        p = -1
        if not(i % 2):
            p = 1
        m = kx_list[i//2]
        n = ky_list[i//2]
        mode_basis_cube[i] = aperture.shaped * fourier_nm(m, n,
                                    pupil_size, p)
        params['m'].append(m)
        params['n'].append(n)
        params['p'].append(p)

    modeBasis = hcipy.ModeBasis([mode.flatten() for mode in mode_basis_cube], pupil_grid)

    return params, modeBasis