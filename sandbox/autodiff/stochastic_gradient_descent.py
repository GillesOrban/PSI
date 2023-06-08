'''
HISTORY
    - 2022/01/02 (GOX): copied from ML_WFS private github repo

'''
import time
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import numpy as np
from numpy.linalg import pinv
import aotools

def stochastic_gradient_descent(init_phase, pupil_aper, target_psf_in,
                                phase_diversity, target_psf_out,
                                num_iters, padding_factor, phase_as_zernCoeffs=False,
                                photon_flux=None,
                                roi_res=None, phase_path=None,
                                loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0,
                                writer=None, dtype=torch.float32, precomputed_H=None,
                               return_metric=False, true_phase=None, nZern=20):
    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------

        true_phase : only use to evaluate the metric

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """

    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)

    # phase at the pupil plane
    pupil_phase = init_phase.requires_grad_(True)

    # optimization variables and adam optimizer
    optvars = [{'params': pupil_phase}]
    if lr_s > 0:
        optvars += [{'params': s, 'lr': lr_s}]
    # optimizer = optim.Adam(optvars, lr=lr, weight_decay=1e-6, eps=1e-5, amsgrad=False)
    # optimizer = optim.Adam(optvars, lr=lr, eps=1e-6, amsgrad=True)
    optimizer = optim.Adam(optvars, lr=lr,  amsgrad=True) #, weight_decay=1e-9)

    if phase_as_zernCoeffs:
        # building zernike basis
        zernBasis = aotools.zernikeArray(pupil_phase.shape[0] + 1, pupil_aper.shape[0])
        zernBasis = torch.Tensor(zernBasis[1:])
        div_vect = torch.Tensor(np.array(pupil_phase.shape[0]))
        div_vect[3] = phase_diversity

    # crop target roi
    target_psf_in = utils.crop_image(target_psf_in, roi_res, stacked_complex=False)
    target_psf_out = utils.crop_image(target_psf_out, roi_res, stacked_complex=False)

    otf_in = torch.abs(torch.fft.fftshift(torch.fft.fft2(target_psf_in)))
    otf_out = torch.abs(torch.fft.fftshift(torch.fft.fft2(target_psf_out)))
    otf_in /= target_psf_in.sum()
    otf_out /= target_psf_out.sum()

    metrics = []
    metrics_zern=[]
    if return_metric:
        nx_size = target_psf_in.shape[0]
        zernike_basis = aotools.zernikeArray(nZern + 1, nx_size, norm='rms')
        ap = zernike_basis[0]
        idx = np.where(ap != 0)
        Z=[]
        for i in range(zernike_basis.shape[0]-1):
            Z.append(zernike_basis[i+1][idx])
        Zinv = torch.Tensor(pinv(Z))

    # run the iterative algorithm
    timer = 0
    for k in range(num_iters):
        print(k)
        t0 = time.time()
        optimizer.zero_grad()
        # forward propagation from the SLM plane to the target plane
        # real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        # slm_field = torch.complex(real, imag)

        # recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
        #                                     prop_model, dtype, precomputed_H)
        #
        # # get amplitude
        # recon_amp = recon_field.abs()
        #
        # # crop roi
        # recon_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)
        if not(phase_as_zernCoeffs):
            recon_psf_in = utils.propagate_field(pupil_phase, pupil_aper,
                                                 padding_factor=padding_factor,
                                                 photon_flux=photon_flux)
            recon_psf_out = utils.propagate_field(pupil_phase + phase_diversity,
                                                  pupil_aper,
                                                  padding_factor=padding_factor,
                                                  photon_flux=photon_flux)
        else:
            recon_psf_in = utils.propagate_field_with_zernike(pupil_aper,
                                                              pupil_phase,
                                                              zernBasis=zernBasis,
                                                              padding_factor=padding_factor,
                                                              photon_flux=photon_flux)

            recon_psf_out = utils.propagate_field_with_zernike(pupil_aper,
                                                               pupil_phase + div_vect,
                                                               padding_factor=padding_factor,
                                                               photon_flux=photon_flux)

        # calculate loss and backprop
        input = torch.Tensor(np.ones((2, target_psf_in.shape[0], target_psf_in.shape[1])))
        target = torch.Tensor(np.ones((2, target_psf_in.shape[0], target_psf_in.shape[1])))
        input[0] = torch.sqrt(recon_psf_in)
        input[1] = torch.sqrt(recon_psf_out)
        target[0] = torch.sqrt(target_psf_in)
        target[1] = torch.sqrt(target_psf_out)
        # input[0] = (recon_psf_in)
        # input[1] = (recon_psf_out)
        # target[0] = (target_psf_in)
        # target[1] = (target_psf_out)
        lossValue = loss(s * input[1], target[1]) + loss(s * input[0], target[0])

        # otf_recon_in = torch.abs(torch.fft.fftshift(torch.fft.fft2(recon_psf_in)))
        # otf_recon_out = torch.abs(torch.fft.fftshift(torch.fft.fft2(recon_psf_out)))
        # otf_recon_in /= recon_psf_in.sum()
        # otf_recon_out /= recon_psf_out.sum()
        # lossValue = loss(s * otf_recon_in, otf_in) + loss(s * otf_recon_out, otf_out)

        lossValue.backward()
        optimizer.step()

        t1 = time.time()
        timer += (t1 - t0)
        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        # with torch.no_grad():
        #     if k % 50 == 0:
        #         utils.write_sgd_summary(slm_phase, out_amp, target_amp, k,
        #                                 writer=writer, path=phase_path, s=s, prefix='test')

        # -- Metrics
        if return_metric:
            unwrap_phase_error = torch.angle(torch.exp(1j * (true_phase - pupil_phase)))
            error = torch.std(unwrap_phase_error[idx])

            coeffs = np.dot(unwrap_phase_error[idx].detach().numpy(), Zinv.detach().numpy())
            wfe_zern = np.sqrt(np.sum(coeffs**2))
            metrics_zern.append(wfe_zern)

            metrics.append((timer, error))

    if return_metric:
        return pupil_phase, metrics, metrics_zern

    else:
        return pupil_phase


class SGD(nn.Module):
    def __init__(self, aperture, phase_diversity,
                 padding_factor, num_iters, useZernCoeffs=False,
                 loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0,
                 device=torch.device('cuda')):
        super(SGD, self).__init__()
        self.aperture = aperture
        self.phase_diversity = phase_diversity
        self.padding_factor = padding_factor
        self.num_iters = num_iters
        self.lr = lr
        self.lr_s = lr_s
        self.init_scale = s0
        self.loss = loss.to(device)
        self.use_zernCoeffs = useZernCoeffs

    def forward(self, target_psf_in, target_psf_out,
                init_phase=None, photon_flux=None,
               return_metric=False, true_phase=None, nZern=20):
        if return_metric is False:
            final_phase = stochastic_gradient_descent(init_phase, self.aperture,
                                                      target_psf_in,
                                                      self.phase_diversity,
                                                      target_psf_out,
                                                      self.num_iters,
                                                      self.padding_factor,
                                                      phase_as_zernCoeffs=self.use_zernCoeffs,
                                                      # photon_flux=photon_flux,
                                                      roi_res=None, phase_path=None,
                                                      loss=self.loss, lr=self.lr,
                                                      lr_s=self.lr_s,
                                                      s0=self.init_scale,
                                                      return_metric = False,
                                                      nZern=nZern)

            return final_phase
        else:
            final_phase, metric, metric_zern = stochastic_gradient_descent(init_phase, self.aperture,
                                                      target_psf_in,
                                                      self.phase_diversity,
                                                      target_psf_out,
                                                      self.num_iters,
                                                      self.padding_factor,
                                                      phase_as_zernCoeffs=self.use_zernCoeffs,
                                                      # photon_flux=photon_flux,
                                                      roi_res=None, phase_path=None,
                                                      loss=self.loss, lr=self.lr,
                                                      lr_s=self.lr_s,
                                                      s0=self.init_scale,
                                                      return_metric = True,
                                                      true_phase=true_phase,
                                                      nZern=nZern)

            return final_phase, metric, metric_zern


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.visualization import imshow_norm,\
        SqrtStretch, MinMaxInterval, PercentileInterval, ManualInterval,\
        LinearStretch, SinhStretch, LogStretch
    import aotools

    verbose = True
    direct = True
    photon_flux = 1e6
    nGrid = 16
    nZern = 20
    pad = 1.7689 #4
    zernBasis = aotools.zernikeArray(nZern + 1, nGrid)
    aperture = torch.Tensor(zernBasis[0])
    zernBasis = torch.Tensor(zernBasis[1:])

    if not direct:

        # -- working with zernike coefficients
        div_coeffs = 2 * np.pi / 4  # lbda/4 rms focus
        # phase_init = torch.Tensor(aberration * 0.5)
        zernCoeffs = np.zeros(nZern)
        zernCoeffs[3] = 0.75
        zernCoeffs[5] = 0.25
        target_psf_in = utils.propagate_field_with_zernike(aperture,
                                                           zernCoeffs,
                                                           zernBasis=zernBasis,
                                                           padding_factor=pad,
                                                           photon_flux=photon_flux)
        zernCoeffs_div = np.copy(zernCoeffs)
        zernCoeffs_div[2] += div_coeffs

        target_psf_out = utils.propagate_field_with_zernike(aperture,
                                                            zernCoeffs_div,
                                                            zernBasis=zernBasis,
                                                            padding_factor=pad,
                                                            photon_flux=photon_flux)
        # reconstruct aberration for plotting purpose
        zernCoeffs = torch.Tensor(zernCoeffs)
        zernCoeffs = torch.reshape(zernCoeffs, (zernCoeffs.shape[0], 1, 1))
        aberration = (zernCoeffs * zernBasis).sum(0, keepdim=True).squeeze(0)
    else:
        # -- working directly with phase map
        diversity = zernBasis[2] * 2 * np.pi / 4  # lbda/4 rms focus
        phase_init = aperture * torch.Tensor((2 * np.random.random((nGrid, nGrid)) - 1))
        aberration = zernBasis[3] * 0.75 + zernBasis[5] * 0.25
        target_psf_in = utils.propagate_field(aberration, aperture,
                                              padding_factor=pad,
                                              photon_flux=photon_flux)
        target_psf_out = utils.propagate_field(aberration + diversity, aperture,
                                               padding_factor=pad,
                                               photon_flux=photon_flux)

    if verbose:
        myStretch = LogStretch()
        percentil = 100
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        im, _ = imshow_norm(target_psf_in, plt.gca(), stretch=myStretch,
                            interval=PercentileInterval(percentil))
        plt.gca().set_axis_off()
        plt.colorbar(im, pad=0.01)

        plt.subplot(122)
        im, _ = imshow_norm(target_psf_out, plt.gca(), stretch=myStretch,
                            interval=PercentileInterval(percentil))
        plt.gca().set_axis_off()
        plt.colorbar(im, pad=0.01)

    if not direct:
        num_iters = 61
        phase_sgd = SGD(torch.Tensor(aperture), torch.Tensor([div_coeffs]),
                        pad, num_iters, useZernCoeffs=True,
                        loss=nn.MSELoss(), lr=0.1, lr_s=0, s0=1.0,
                        device=torch.device('cpu'))

        initCoeffs = torch.Tensor((2*np.random.random(nZern) - 1) * 0.1)
        final_coeffs = phase_sgd(target_psf_in, target_psf_out,
                                 init_phase=initCoeffs, photon_flux=None)
        check_psf = utils.propagate_field_with_zernike(aperture, final_coeffs,
                                                       padding_factor=pad,
                                                       photon_flux=photon_flux)
        # reconstruct final phase
        final_coeffs = torch.Tensor(final_coeffs)
        final_coeffs = torch.reshape(final_coeffs, (final_coeffs.shape[0], 1, 1))
        final_phase = (final_coeffs * zernBasis).sum(0, keepdim=True).squeeze(0)
    else:
        # -- working directly with phase map
        num_iters = 51
        phase_sgd = SGD(torch.Tensor(aperture), torch.Tensor(diversity),
                        pad, num_iters,
                        loss=nn.MSELoss(), lr=0.2, lr_s=0, s0=1.0,
                        device=torch.device('cpu'))

        final_phase = phase_sgd(target_psf_in, target_psf_out,
                                init_phase=phase_init*0, photon_flux=None)
        check_psf = utils.propagate_field(final_phase.detach().numpy(), aperture,
                                          padding_factor=pad,
                                          photon_flux=photon_flux)

    # --- Compute statistics ---
    # Build Zinv
    zernBasis = zernBasis.detach().numpy()
    idx = np.where(aperture != 0)
    Z = []
    for i in range(zernBasis.shape[0]-1):
        Z.append(zernBasis[i][idx])
    Zinv = pinv(Z)

    aperture = aperture.detach().numpy()
    aberration = aberration.detach().numpy()
    resPhase = aberration - final_phase.detach().numpy()
    resWFE = np.std(resPhase[aperture == 1])
    print('Input WFE = {0:.1f}nm'.format(np.std(aberration[aperture == 1]) * 2200 / (2*np.pi)))
    print('Residual WFE = {0:.1f}nm'.format(resWFE * 2200 / (2*np.pi)))
    coeffs = np.dot(resPhase[idx], Zinv)
    wfe_zern = np.sqrt(np.sum(coeffs**2))
    print('Projected WFE = {0:.1f}nm'.format(wfe_zern * 2200 / (2*np.pi)))

    if verbose:
        # myInterval = PercentileInterval(percentil)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(aberration, vmin=aberration.min(), vmax=aberration.max())
        plt.colorbar(pad=0.01)

        plt.subplot(122)
        plt.imshow(final_phase.detach().numpy(), vmin=aberration.min(), vmax=aberration.max())
        plt.colorbar(pad=0.01)

        myStretch = LogStretch()
        percentil = 100
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        im, _ = imshow_norm(check_psf.detach().numpy(), plt.gca(), stretch=myStretch,
                            interval=PercentileInterval(percentil))
        plt.gca().set_axis_off()
        plt.colorbar(im, pad=0.01)
