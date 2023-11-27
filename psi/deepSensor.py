

import hcipy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import psi.psi_utils as psi_utils
from .configParser import loadConfiguration
from .instruments import  CompassSimInstrument, DemoCompassSimInstrument, HcipySimInstrument
from .helperFunctions import LazyLogger, timeit, build_directory_name, \
    copy_cfgFileToDir, dump_config_to_text_file

from astropy.visualization import imshow_norm,\
    SqrtStretch, MinMaxInterval, PercentileInterval, \
    LinearStretch, SinhStretch, LogStretch, ManualInterval

from psi.deep_wfs.dataset_generator import dataGen
from psi.deep_wfs.training_model import dataTrain
from psi.deep_wfs.inference import dataInfer



class DeepSensor():

    def __init__(self, config_file, logger=LazyLogger('DEEP')):
        self.logger=logger
        self.logger.info('Loading and checking configuration')
        self._config_file = config_file
        self.cfg = loadConfiguration(config_file)

        self.generator = dataGen()  # empty object -- need setting up
        self.trainer = dataTrain()
        self.evaluator = dataInfer()

    def setup(self):
        '''
        
        
        '''
        self.logger.info('Initialize the instrument object & building the optical model')
        self.inst = eval(self.cfg.params.instrument)(self.cfg.params)
        self.inst.build_optical_model()

        # TODO Check specific attribute

        # TODO If model keyword is given and file exists, load CNN model
        self.evaluator.setup(self.cfg.params.f_inference)

        # Initialize modal basis
        self._initModalBasis()

        # Initialize logging buffer
        self.iter = 0

        # Plotting & saving results


    def _initModalBasis(self, nbOfModes=20):
        self.logger.info('Initializing DeepSensor modal basis')
        diam = 1
        radial_cutoff = False
        nmode_shift = 3
        self.M2C = hcipy.make_zernike_basis(nbOfModes + nmode_shift, diam,
                                            self.inst.pupilGrid, 1,
                                            radial_cutoff=radial_cutoff)

        self.M2C = psi_utils.reorthonormalize(self.M2C,
                                              self.inst.aperture)
        self.M2C_matrix = self.M2C.transformation_matrix[:, nmode_shift:]
        self.C2M =hcipy.inverse_tikhonov(self.M2C.transformation_matrix,
                                         1e-3)[nmode_shift:,:]


    def buildModel(self):
        '''
        - Generate dataset ?
        - Training 
        - Preparing for future inference
        '''
        self.logger.info('Generating dataset')
        # If data set do not exists
        self.generator.setup(self.inst, self.C2M)
        tag_name = 'toto'
        self.generator.genData(tag_name)

        # If ok, continue and train
        self.logger.info('Starting training')
        self.trainer.setup()
        # Ask user to continue ?
        inp = input('Would you like to start training the model ? (q to quit)')
        if inp == 'q':
            return 0
        
        self.trainer.trainModel()





    def next(self, leak=1, integrator=True):
        # TODO replace psi_framerate by kernel_framerate
        nbOfSeconds = 1/self.cfg.params.psi_framerate
        science_images_buffer = self.inst.grabScienceImages(nbOfSeconds)
   
        self.science_image = science_images_buffer.mean(0)

        self._modes = self.evaluator.infer(self.science_image[np.newaxis,:,:]).squeeze()

        self._wavefront = self.M2C_matrix.dot(self._modes)
        # _dim = self.inst.pupilGrid.shape[0]
        # self._wavefront = np.reshape(_wavefront, (_dim, _dim))
        self.inst.setNcpaCorrection(-self._wavefront,
                                    phase_prop=self.cfg.params.gain_P * \
                                        self._wavefront,
                                    integrator=True,
                                    leak=leak)
        self.iter +=1

        # if display:
        #     self.show()

    def loop(self):
        pass

    def evaluateSensorEstimate(self):
        pass

    def show(self):
        ax1 = plt.subplot(141, label='science')
        im1, _= imshow_norm(self.science_image, stretch=LogStretch(), ax=ax1)
        vmin = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 1)
        vmax = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 99)
        inter = ManualInterval(vmin, vmax)
        #--
        ax2 = plt.subplot(142, label='dist')
        im2, _=imshow_norm((self.inst.phase_wv + self.inst.phase_ncpa).shaped,
                           interval=inter, ax=ax2)
        #--
        ax3 = plt.subplot(143, label='wfs')
        _dim = self.inst.pupilGrid.shape[0]
        # im3, _=imshow_norm(-self.inst.phase_ncpa_correction.reshape((_dim, _dim)) * \
        #                    self.inst.aperture.shaped,
        #                    interval=inter, ax=ax3)
        im3, _=imshow_norm(self._wavefront.shaped * \
                           self.inst.aperture.shaped,
                           interval=inter, ax=ax3)
        #--
        ax4 = plt.subplot(144, label='res')
        im4, _=imshow_norm(self.inst.aperture.shaped *
                           (self.inst.phase_wv + self.inst.phase_ncpa +
                           self.inst.phase_ncpa_correction).shaped,
                           interval=inter, ax=ax4)

        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax4.set_axis_off()
        ax2.set_title('static NCPA + WV')
        ax3.set_title('NCPA correction')
        ax4.set_title('Residuals')
        plt.tight_layout()
        # plt.draw()
        # plt.pause(0.01)
        # time.sleep(0.1)
        # self._ims.append([im1, im2, im3, im4])        