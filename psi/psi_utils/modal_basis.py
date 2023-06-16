import numpy as np
import hcipy
import aotools
from astropy.modeling import models

def reorthonormalize(modal_basis, aperture):
    # modal_basis.transformation_matrix can contain NaN in some cases...
    #   hopefully there should be masked by the aperture
    modal_basis.transformation_matrix[np.isnan(modal_basis.transformation_matrix)]=0

    transformation_matrix = modal_basis.transformation_matrix * aperture[:, np.newaxis]
    transformation_matrix[np.isnan(transformation_matrix)] = 0
    # q, r = np.linalg.qr(transformation_matrix, mode='complete')
    
    nvalid_pixels = np.sum(aperture)
    cross_product = 1 / nvalid_pixels * np.dot(transformation_matrix.T, transformation_matrix)
    L = np.linalg.cholesky(cross_product)

    basis_orthonormalized = np.dot(transformation_matrix, np.linalg.inv(L).T)

    return hcipy.ModeBasis(basis_orthonormalized, modal_basis.grid)

def gendrinou_basis(pupil_grid, aperture, nActOnDiam):
    '''
        Build a 'Gendrinou' basis (see email 14/10/2020). 
        Based on actuator inter-distance

        To map from the actuator grid (assumed to be cartesian) to the pupil grid, 
        we use Gaussian influence functions

        Parameters
        ----------
        pupil grid : hcipy CartesianGrid
        aperture   : hcipy field or 2D numpy array
        nActOnDiam  : number of actuators along the aperture diameter. 
            Sets the number of modes in the basis (2 * pi * (nActOnDiam/2)**2)

        Returns
        -------
        modeBasis:  hcipy.ModeBasis
            computed on the pupil_grid and with the aperture mask applied
    '''
    nGrid = pupil_grid.dims[0]

    x = np.linspace(-nActOnDiam/2, nActOnDiam/2, nActOnDiam) #np.arange(nActOnDiam) +0.5 - nActOnDiam//2 #np.linspace(- nGrid / 2, nGrid/2 , nActOnDiam +1) + 0.5
    xx, yy = np.meshgrid(x,x)

    pupil_dm = aotools.circle(nActOnDiam/2, nActOnDiam)

    xxf = xx[pupil_dm==1]
    yyf = yy[pupil_dm==1]
    nAct = int(np.sum(pupil_dm))
    A = np.zeros((nAct, nAct))
    # --- Gendrinou recipe ---
    for i in range(nAct):
            A[i, :] = np.sqrt((xxf - xxf[i])**2 + (yyf - yyf[i])**2)

    A = A**(5/3)
    # Piston removal
    P = np.identity(nAct) - (1/nAct)*np.ones((nAct, nAct))
    Ap = np.dot(P, np.dot(A, np.transpose(P)))

    U, S, V = np.linalg.svd(Ap)

    # -- Generating pseudo-influence functions to map on a larger pupil grid --
    coords = np.linspace(-nActOnDiam/2, nActOnDiam/2, nGrid ) 
    xgrid, ygrid = np.meshgrid(coords,coords)
    # pupil = aotools.circle(nGrid/2, nGrid )

    gaussian = models.Gaussian2D(amplitude=1,
                                    x_mean=0,
                                    y_mean=0,
                                    x_stddev=1, #nGrid/nActOnDiam/2,
                                    y_stddev=1, #nGrid/nActOnDiam/2,
                                    theta=0)

    IFs = np.zeros((nAct, (nGrid ) * (nGrid )))
    for i in range(nAct):
        IF = gaussian(xgrid - xxf[i], ygrid - yyf[i])
        IFs[i] = IF.flatten()

    # Remap to 2D
    if type(aperture) == hcipy.Field:
        aperture = aperture.shaped

    modeBasis = np.zeros((nAct, nGrid, nGrid))
    cmdBasis = np.zeros((nAct, nActOnDiam, nActOnDiam))
    for i in range(nAct):
        cmdBasis[i][pupil_dm==1] = U[:, i] #Up[:,i]
        modeBasis[i] = np.reshape(np.sum(IFs.T * U[:,i], axis=1), (nGrid, nGrid)) * aperture 
        
        norm = np.std(modeBasis[i][aperture==1])
        modeBasis[i] /= norm
        cmdBasis[i] /= norm

    modeBasis_hcipy = hcipy.ModeBasis([modeBasis[i].flatten() for i in range(nAct)], pupil_grid)
    # optional: return cmdBasis)

    return modeBasis, modeBasis_hcipy

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


    if type(aperture) == hcipy.Field:
        aperture = aperture.shaped

    for i in range(kxy_list.shape[0]*2):
        p = -1
        if not(i % 2):
            p = 1
        m = kx_list[i//2]
        n = ky_list[i//2]

        mode_basis_cube[i] = aperture * fourier_nm(m, n,
                                    pupil_size, p)
        params['m'].append(m)
        params['n'].append(n)
        params['p'].append(p)

    modeBasis = hcipy.ModeBasis([mode.flatten() for mode in mode_basis_cube], pupil_grid)

    return params, modeBasis

# def 
#         self.smallGrid = hcipy.make_pupil_grid(self.cfg.params.asym_nsteps - 1)
#         self.M2C_small = hcipy.make_zernike_basis(nbOfModes, diam,
#                                                    self.smallGrid,
#                                                   4,
#                                                   radial_cutoff=radial_cutoff)
#         self.M2C_small = psi_utils.reorthonormalize(self.M2C_small, self._small_aperture.flatten())
#         self.C2M_small = hcipy.inverse_tikhonov(self.M2C_small.transformation_matrix, 1e-3)

#         self.M2C_large = hcipy.make_zernike_basis(nbOfModes, diam,
#                                                   self.inst.pupilGrid, 4,
#                                                   radial_cutoff=radial_cutoff)
