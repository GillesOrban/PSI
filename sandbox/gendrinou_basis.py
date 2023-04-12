import numpy as np
import aotools
import matplotlib.pyplot as plt
from astropy.visualization import imshow_norm, SqrtStretch, MinMaxInterval
from astropy.modeling import models

nGrid = 256
nActOnDiam=16
x = np.linspace(-nActOnDiam/2, nActOnDiam/2, nActOnDiam) #np.arange(nActOnDiam) +0.5 - nActOnDiam//2 #np.linspace(- nGrid / 2, nGrid/2 , nActOnDiam +1) + 0.5
xx, yy = np.meshgrid(x,x)

pupil_dm = aotools.circle(nActOnDiam/2, nActOnDiam)

# xxf  = xx.flatten()
# yyf = yy.flatten()
xxf = xx[pupil_dm==1]
yyf = yy[pupil_dm==1]
nAct = int(np.sum(pupil_dm))
A = np.zeros((nAct, nAct))

for i in range(nAct):
        A[i, :] = np.sqrt((xxf - xxf[i])**2 + (yyf - yyf[i])**2)

A = A**(5/3)
# Piston removal
# P = np.identity(nAct) - (1/nAct)*np.ones((nAct, nAct))
# Ap = np.dot(P, np.dot(A, np.transpose(P)))
Ap = A
U, S, V = np.linalg.svd(Ap)

# Generating pseudo-influence functions to map on a larger pupil grid
coords = np.linspace(-nActOnDiam/2, nActOnDiam/2, nGrid ) 
xgrid, ygrid = np.meshgrid(coords,coords)
pupil = aotools.circle(nGrid/2, nGrid )

gaussian = models.Gaussian2D(amplitude=1,
                                x_mean=0,
                                y_mean=0,
                                x_stddev=1, #nGrid/nActOnDiam/8,
                                y_stddev=1, #nGrid/nActOnDiam/8,
                                theta=0)

IFs = np.zeros((nAct, (nGrid ) * (nGrid )))
norm = np.sum(gaussian(xgrid, ygrid).flatten())
for i in range(nAct):
     IF = gaussian(xgrid - xxf[i], ygrid - yyf[i])
     IFs[i] = (IF.flatten() / norm)


# Remap to 2D
modeBasis = np.zeros((nAct, nGrid, nGrid))
cmdBasis = np.zeros((nAct, nActOnDiam, nActOnDiam))
for i in range(nAct):
    cmdBasis[i][pupil_dm==1] = U[:, i] #Up[:,i]
    modeBasis[i] = np.reshape(np.sum(IFs.T * U[:,i], axis=1), (nGrid, nGrid)) * pupil #np.dot(IFs.T, Up[:, i]), (33, 33))
    
    norm = np.std(modeBasis[i][pupil==1])
    modeBasis[i] /= norm
    cmdBasis[i] /= norm



# Compute PSF
def calculatePSF(Efield, pupil, crop_size=32, pad_fac=1, norm=True):
    padding = Efield.shape[0]
    padding_left = padding * pad_fac #+ 1
    padding_right = padding * pad_fac
    pupil = np.pad(pupil, (padding_left, padding_right), mode='constant',
                   constant_values=0)
    Efield = np.pad(Efield, (padding_left, padding_right), mode='constant',
                       constant_values=0)
    h = pupil * Efield #np.exp(1j * wavefront)
    H = aotools.ft2(h, 1)
    
    cc = Efield.shape[0] // 2
    psf= (np.abs(H)**2)[cc - crop_size//2: cc + crop_size//2,
                        cc - crop_size//2: cc + crop_size//2]
    if norm:
        psf /= np.max(psf)
    return psf

def show(idx=5, amp=3):
    phase = amp * modeBasis[idx]
    psf = calculatePSF(np.exp(1j*phase), pupil, crop_size=64, pad_fac=2)

    plt.figure()
    plt.subplot(121)
    plt.imshow(phase)
    plt.subplot(122)
    imshow_norm(psf, stretch=SqrtStretch(), interval=MinMaxInterval())