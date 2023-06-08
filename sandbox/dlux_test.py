# Core jax
import jax
import jax.numpy as np
import jax.random as jr

# Optimisation
import equinox as eqx
import optax

# Optics
import dLux as dl

# Plotting/visualisation
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tqdm.notebook import tqdm

# %matplotlib inline
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120


# Basic Optical Parameters
diameter = 1.
wf_npix = 256 

# Detector Parameters
det_npix = 256 
det_pixsize = dl.utils.arcseconds_to_radians(1e-2)

# Generate an asymmetry
asymmetry = np.ones((wf_npix,wf_npix)).at[112:144,0:128].set(0)
asymmetric_mask = dl.TransmissiveOptic(asymmetry)

# Define the aberrations
zernikes = np.arange(4, 11)
coeffs = 2e-8*jr.normal(jr.PRNGKey(0), (len(zernikes),))




# Define Optical Configuration
optical_layers = [
    dl.CreateWavefront    (wf_npix, diameter),
    dl.SimpleAperture(wf_npix, nstruts=3, secondary_ratio=0.1, strut_ratio=0.01, 
                      zernikes=zernikes, coefficients=coeffs),
    dl.TransmissiveOptic  (asymmetry),
    dl.NormaliseWavefront (),
    dl.AngularMFT         (det_npix, det_pixsize)]



# Create a point source
wavels = np.linspace(450e-9, 550e-9, 5)
source = dl.PointSource(wavelengths=wavels)

# Construct the instrument with the source
telescope = dl.Instrument(optical_layers=optical_layers, sources=[source])

# Get aberrations
aberr_in = telescope.CircularAperture.get_opd()
aper_in = telescope.CircularAperture.get_aperture()

# Get mask, setting nan values for visualisation
mask = 1.0*(aper_in*asymmetry > 1e-5) # for visualization, have a nan mask
mask = mask.at[mask==False].set(np.nan)
cmap = get_cmap("inferno")
cmap.set_bad('k',1.)

# Model the PSF using the .model() method
psf = telescope.model()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Aberrations")
plt.imshow(mask*aberr_in*1e9, cmap=cmap)
plt.colorbar(label='nm')

plt.subplot(1, 2, 2)
plt.title("PSF")
plt.imshow(psf**0.5)
plt.colorbar()
plt.show()

#######

# Define path to the zernikes
zernikes = 'CircularAperture.coefficients'

# Generate new random set of zernikes
coeffs_init = 2e-8*jr.normal(jr.PRNGKey(1), [len(coeffs)])

# Generate a new model with updated zernike coefficients
model = telescope.set(zernikes, coeffs_init)


args = model.get_args(zernikes)
@eqx.filter_jit
@eqx.filter_value_and_grad(arg=args)
def loss_func(model, data):
    out = model.model()
    return -np.sum(jax.scipy.stats.poisson.logpmf(data, out))

%%time
loss, initial_grads = loss_func(model, psf) # Compile
print("Initial Loss: {}".format(loss))


%%timeit
loss = loss_func(model, psf)[0].block_until_ready()



optim = optax.adam(2e-9)
opt_state = optim.init(model)

losses, models_out = [], []
with tqdm(range(100),desc='Gradient Descent') as t:
    for i in t: 
        # calculate the loss and gradient
        loss, grads = loss_func(model, psf) 
        
        # apply the update
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        # save results
        models_out.append(model) 
        losses.append(loss)
        
        t.set_description('Loss %.5f' % (loss)) # update the progress bar
final_psf = model.model(source=source)