import torch
from Autodiff_instrument import ADInstrument

from psi.configParser import loadConfiguration
import matplotlib.pyplot as plt
import hcipy
import numpy as np

config_file = 'config_ADsim.py'
cfg = loadConfiguration(config_file)
inst = ADInstrument(cfg.params)
inst.build_optical_model()
inst.photon_flux = 1e10
inst.compute_jacobian(plane='img')

init_aperture = inst.aperture_torch * 1
# init_aperture = hcipy.aperture.make_circular_aperture(inst.diam)(inst.pupilGrid)
# init_aperture = torch.Tensor(np.array(init_aperture.shaped))

inst.aperture_torch = torch.rand(init_aperture.shape)
# define optimizer
# def aperture_param(aper):
#     aper.requires_grad_(True)
#     m = torch.nn.Threshold(0,0, inplace=False)
#     m2 = torch.nn.Threshold(-1,-1 , inplace=False)

#     toto = m(inst.aperture_torch)
#     apo  = - m2(-toto)
#     return torch.nn.Parameter(apo)

inst.aperture_torch.requires_grad_(True)
optvars = [{'params': inst.aperture_torch}]

optimizer = torch.optim.Adam(optvars,
                             lr=0.1, amsgrad=True, weight_decay=0.)
# optimizer = torch.optim.SGD(optvars,
#                              lr=0.5)

def threshold_aper(aper, sigmoid=True):
    if sigmoid is False:
        m = torch.nn.Threshold(0,0 , inplace=False)
        m2 = torch.nn.Threshold(-1,-1 , inplace=False)

        toto = m(aper)
        return - m2(-toto)
    else:
        return torch.sigmoid(inst.aperture_torch)
# model
def model(inst, coeffs, plane='img'):

    apo  = threshold_aper(inst.aperture_torch) #- m2(-toto)
    # apo = torch.sigmoid(inst.aperture_torch)

    aperture = apo * init_aperture
    if inst._inst_mode == 'ELT':
        inst.optical_model.setPupilStop(aperture)
    elif inst._inst_mode == 'CVC':
        inst.optical_model.setLyotStop(aperture)


    # inst.compute_jacobian(plane=plane)
    if plane is 'img':
        meas_2d = inst.propagate_forward_with_zernike(coeffs, True) -\
            inst.propagate_forward_with_zernike(coeffs*0, False)
        meas = meas_2d.flatten()
    elif plane is 'fourier':
        meas_2d = inst.propagate_to_fourier_plane_with_zernike(coeffs, True)
        meas = meas_2d[torch.isnan(meas_2d)==False]

    c_retrieved = torch.matmul(inst.jacobian_matrix_inverse, meas).type(torch.float)

    return inst, c_retrieved

def criterion(est, true, aper):
    wfe =  torch.nn.MSELoss(reduction='sum')(est, true)
    print(wfe)
    flux = 1e-4 * (torch.sum(init_aperture)- torch.sum(aper)) / torch.sum(init_aperture)
    # flux = 1e11 / torch.sum(init_aperture) * torch.nn.L1Loss()(init_aperture, aper)
    print(flux)
    return wfe + flux

def criterion2(init, aper):

    flux = 1e2 * (torch.sum(init)- torch.sum(aper)) / torch.sum(init)
    print(flux)
    return flux


# criterion = torch.nn.MSELoss(reduction='sum')

coeffs = torch.eye(inst.nmodes) * 0.1
coeffs = coeffs[8:10, :]
# data = [model(coeffs[i]) for i in range(coeffs.shape[0])]

loss_vec = []
n_iter = 3
nbatch = 10

for i in range(n_iter):
    print("iter = {0}".format(i))
    for j in range(coeffs.shape[0]):
        print("j = {0}".format(j))
        loss = 0
        for t in range(nbatch):
            inst, c_hat = model(inst, coeffs[j])
            loss += criterion(c_hat, coeffs[j], threshold_aper(inst.aperture_torch) * init_aperture) #torch.sigmoid(inst.aperture_torch)*init_aperture)
            
        loss_vec.append(loss.item())
        optimizer.zero_grad()
        loss.retain_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        inst.compute_jacobian(plane='img')


        # loss2 = criterion2(init_aperture,threshold_aper(inst.aperture_torch) * init_aperture )
        # # loss2 = criterion2(init_aperture,torch.sigmoid(inst.aperture_torch)*init_aperture )

        # optimizer.zero_grad()
        # # loss.retain_grad()
        # loss2.backward()
        # optimizer.step()




# Test
inst.calculate_ref_psf()

coeffs = torch.eye(inst.nmodes) * 0.1
inst.test_wavefront_retrieval(coeffs[9], plane='img')

plt.figure()
plt.subplot(121)
plt.imshow(init_aperture.detach().numpy())
plt.subplot(122)
plt.imshow((threshold_aper(inst.aperture_torch) * init_aperture).detach().numpy())