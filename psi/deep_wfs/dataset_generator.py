import os
import numpy as np
import h5py
from tqdm import tqdm
import psi.deep_wfs.utils.read_data  as rt
from psi.helperFunctions import LazyLogger
import psi.psi_utils as psi_utils
import astropy.io.fits as fits

# config_file='config/config_deep_learning.py'
# deep_sensor = DeepSensor(config_file)
# deep_sensor.setup()


class dataGen():

    def __init__(self, logger=LazyLogger('deep_gen')):
        self.logger = logger
        self.scaling_ps = 1

    def setup(self,  inst, C2M, sensor_params=None, conf_file=None):

        
        self._inst = inst  
        self._C2M = C2M

        extra={'zernike_unit': 'rad',
               'defocus': 0,
               'wavelength': self._inst.wavelength * 1e9,   # nm
               'nb_modes': self._C2M.shape[0],
               'channels': 1,
               'config_simulation' : False}
        if sensor_params is not None:
            # adding the sensor/simulation configuration as a dictionary
            sensor_config = vars(sensor_params).copy()
            extra['config_simulation'] = True
            for key in extra.keys():
                if key in sensor_config.keys():
                    self.logger.warn('Key {0} already in config'.format(key))
                    sensor_config.pop(key)
            extra = {**extra, **sensor_config}

        self.setConfig(conf_file, extra_config=extra)


    def setConfig(self, conf_file=None, extra_config=None):
        if conf_file is None:
            conf_file = os.path.dirname(__file__) + "/config/generator_config.yml"
        else:
            pass
        self.config = rt.read_conf(conf_file=conf_file)

        if extra_config is not None:
            for key, value in extra_config.items():
                if (type(extra_config[key]) in [int, str, float, bool]):
                    self.config[key] = value

    def genData(self, tag_name, store_data=True, phase_fname=''):
        self._inst.include_residual_turbulence = False
        self._inst.include_water_vapour = False
        self._inst.ncpa_dynamic = False
        self._inst.phase_wv *= 0
        self._inst.phase_ncpa *= 0
        

        self.phase_cube = fits.getdata(phase_fname)


        self.config['nb_modes'] = self._C2M.shape[0]

        nmodes = self.config['nb_modes']
        nentries = self.config['nb_samples']
        dim = self._inst.focalGrid.shape[0]
        zernike_coeff = np.zeros((nentries, nmodes))
        psfs = np.zeros((nentries, dim, dim))

        for i in tqdm(range(nentries)):
            # # Grab new images
            # nbOfSeconds = 0.1  # 0.1 seconds is the shortest possible with CompassSim
            # # TODO replace grab by my propagator with random phase screen for higher efficiency
            # science_images_buffer = self._inst.grabScienceImages(nbOfSeconds)

            # science_image = science_images_buffer.mean(0)
            # # crop PSFs ?

            # # Phase screens
            # phase_screen = np.copy(self._inst.phase_wv_integrated)
            phase_screen = self._get_phase_screen(i)
            self._inst.phase_ncpa = phase_screen
            nbOfSeconds = 0.1  # 0.1 seconds is the shortest possible with CompassSim
            science_images_buffer = self._inst.grabScienceImages(nbOfSeconds)
            science_image = science_images_buffer.mean(0)

            # print('toto:')
            # print(self._inst.include_water_vapour)
            
            modes = self._C2M.dot(phase_screen) * self.config['wavelength']/ (2 * np.pi)  # nm

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
    
    def _get_phase_screen(self, idx):
        size_pupil_grid = int(self._inst.pupilGrid.shape[0])
        phase_screen  = self._inst.conv2rad_wv * \
            psi_utils.process_screen(self.phase_cube[idx],
                                size_pupil_grid,
                                self._inst.aperture, rotate=True)
        phase_screen *= self.scaling_ps
        return phase_screen
