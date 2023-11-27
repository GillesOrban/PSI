import os
import numpy as np
import h5py
from tqdm import tqdm
import psi.deep_wfs.utils.read_data  as rt
from psi.helperFunctions import LazyLogger


# config_file='config/config_deep_learning.py'
# deep_sensor = DeepSensor(config_file)
# deep_sensor.setup()

# config={'zernike_unit': 'rad',
#         'defocus': 0,
#         'zernike_seed': 0,
#         'wavelength': 1,
#         'channels': 1,
#         'nb_modes': 0,
#         'nb_samples': 0}


class dataGen():

    def __init__(self, logger=LazyLogger('deep_gen')):
        self.logger = logger


    def setup(self, inst, C2M, conf_file=None):
        config={'zernike_unit': 'rad',
            'defocus': 0,
            'zernike_seed': 0,
            'wavelength': 1,
            'channels': 1,
            'nb_modes': 0,
            'nb_samples': 10}
        
        self.setConfig(conf_file, extra_config=config)
        #self.config = config
        self._inst = inst  
        self._C2M = C2M

    def setConfig(self, conf_file=None, extra_config=None):
        if conf_file is None:
            conf_file = os.path.dirname(__file__) + "/config/generator_config.yml"
        else:
            pass
        self.config = rt.read_conf(conf_file=conf_file)

        if extra_config is not None:
            for key, value in extra_config.items():
                self.config[key] = value

    def genData(self, tag_name, store_data=True):
        self.config['nb_modes'] = self._C2M.shape[0]

        nmodes = self.config['nb_modes']
        nentries = self.config['nb_samples']
        dim = self._inst.focalGrid.shape[0]
        zernike_coeff = np.zeros((nentries, nmodes))
        psfs = np.zeros((nentries, dim, dim))

        for i in tqdm(range(nentries)):
            # Grab new images
            nbOfSeconds = 0.1  # 0.1 seconds is the shortest possible with CompassSim
            # TODO replace grab by my propagator with random phase screen for higher efficiency
            science_images_buffer = self._inst.grabScienceImages(nbOfSeconds)

            science_image = science_images_buffer.mean(0)
            # crop PSFs ?

            # Phase screens
            phase_screen = np.copy(self._inst.phase_wv_integrated)
            modes = self._C2M.dot(phase_screen) # in rad rms ?

            psfs[i] = science_image
            zernike_coeff[i] = modes

        self._psfs = psfs
        self._zernike_coeff = zernike_coeff

        if store_data:
            filename = self.config['dataset_path'] + '/ds_{}.h5'.format(tag_name)
            if self._inst._inst_mode =='IMG':
                asym_stop = self._inst.aperture.shaped
            else:
                asym_stop=self._inst.lyot_stop_mask

            with h5py.File(filename, mode="w") as hdf:
                # Save the Zernike coefficients:
                hdf.create_dataset("zernike_coefficients", data=zernike_coeff)
                # Save the PSFs:
                hdf.create_dataset("psfs_1", data=psfs,
                                    compression="gzip", compression_opts=4)
                
                # Save the aperture:
                hdf.create_dataset("asymmetric_stop", data=asym_stop,
                                    compression="gzip", compression_opts=4)

                # Add attributes:
                hdf.attrs["0"] = 0
                hdf["zernike_coefficients"].attrs['unit'] = self.config["zernike_unit"]
                hdf["psfs_1"].attrs['defocus'] = self.config["defocus"]  # in nm
                # hdf.attrs['seed'] = config["zernike_seed"]
                # hdf.attrs['nb_samples'] = zernike_coeff.shape[0]
                # hdf.attrs['aperture'] = aperture
                hdf.attrs.update(self.config)
    
    # def _save_to_file()

# # Save to hdf5 file
# filename='./toto.h5'
# aperture = inst.aperture.shaped

# # attrs = config
# config['nb_modes'] = nmodes
# config['nb_samples'] = nentries

# with h5py.File(filename, mode="w") as hdf:
#     # Save the Zernike coefficients:
#     hdf.create_dataset("zernike_coefficients", data=zernike_coeff)
#     # Save the PSFs:
#     hdf.create_dataset("psfs_1", data=psfs,
#                         compression="gzip", compression_opts=4)
#     # Add attributes:
#     hdf.attrs["0"] = 0
#     hdf["zernike_coefficients"].attrs['unit'] = config["zernike_unit"]
#     hdf["psfs_1"].attrs['defocus'] = config["defocus"]  # in nm
#     # hdf.attrs['seed'] = config["zernike_seed"]
#     # hdf.attrs['nb_samples'] = zernike_coeff.shape[0]
#     hdf.attrs['aperture'] = aperture
#     hdf.attrs.update(config)

# def read_data(filename, dataset_size):
#     """
#     TODO use readTools  instead

#     Example showing how the hdf5 datasets can be read.
#     """
#     with h5py.File(filename, 'r') as hf:
#         # Putting the dataset as tensors into a dictionary:
#         zern_coeffs = np.array(hf['zernike_coefficients'][:dataset_size])
#         psfs_in = np.array(hf["psfs_1"][ :dataset_size, :, :])
#         # psfs_out = np.array(hf["psfs"][1, :dataset_size, :, :])

#         # db = {}
#         # for key in hf.keys():
#         #     db[key] = hf[key][:]
#         attrs = dict(hf.attrs.items())

#     return attrs, zern_coeffs, psfs_in#, psfs_out