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