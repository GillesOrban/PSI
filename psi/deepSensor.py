
import os
import hcipy
import numpy as np
import pprint
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
from psi.abstract_sensor import AbstractSensor
import psi.deep_wfs.utils.read_data as rt
from colorama import Fore


class DeepSensor(AbstractSensor):

    def __init__(self, config_file, logger=LazyLogger('DEEP')):
        super().__init__()
        self.logger=logger
        self.logger.info('Loading and checking configuration')
        self._config_file = config_file
        self.cfg = loadConfiguration(config_file)

        self.generator = dataGen()  # empty object -- need setting up
        self.trainer = dataTrain()
        self.evaluator = dataInfer()

        self._log_modes_truth=[]
        self._log_modes_meas=[]

    def setup(self):
        '''
        
        
        '''
        self.logger.info('Initialize the instrument object & building the optical model')
        self.inst = eval(self.cfg.params.instrument)(self.cfg.params)
        self.inst.build_optical_model()

        # TODO Check specific attribute

        # TODO If model keyword is given and file exists, load CNN model
        # self.evaluator.setup(self.cfg.params.f_inference)

        # Initialize modal basis
        self._initModalBasis(nbOfModes=self.cfg.params.nb_modes)

        # Initialize logging buffer
        self.iter = 0
        self._ncpa_correction_long_term = 0
        self._loop_stats = []

        # self.wavelength = 1

        # Plotting & saving results
        # -- Plotting & saving results
        if self.cfg.params.save_loop_statistics:
            if self.cfg.params.save_dirname is None or self.cfg.params.save_dirname=='': 
                self._directory = build_directory_name(self._config_file,
                                                self.cfg.params.save_basedir)
            else:
                self._directory = self.cfg.params.save_basedir + self.cfg.params.save_dirname

            if not os.path.exists(self._directory):
                os.makedirs(self._directory)

            # copy initial config file to directory
            copy_cfgFileToDir(self._directory, self._config_file)

            # copy current configuration to text file
            dump_config_to_text_file(self._directory + '/config/' + 'current_config.txt',
                                     self.cfg.params)

            if self.cfg.params.save_phase_screens:
                self._directory_phase = self._directory + 'residualNCPA/'
                os.mkdir(self._directory_phase)

            self.logger.info('Results will be stored in '
                             '{0}'.format(self._directory))

    def _initModalBasis(self, nbOfModes=20, nmode_shift=3, reortho=True):
        self.logger.info('Initializing modal basis with {0} modes'.format(nbOfModes))
        
        aper=None
        if reortho:
            if self.cfg.params.pupil == 'ELT':
                grid_diam = self.inst.pupilGrid.delta[0] * self.inst.pupilGrid.dims[0]
                aper = hcipy.aperture.make_circular_aperture(0.98 * grid_diam)(self.inst.pupilGrid)
                aper -= hcipy.aperture.make_circular_aperture(0.25 * grid_diam)(self.inst.pupilGrid)
                if self.cfg.params.inst_mode == 'IMG' or \
                        self.cfg.params.inst_mode == 'APP' or \
                        self.cfg.params.inst_mode == 'SPP':
                    self.logger.warn('Include asymmetry in dummy pupil to compute M2C and C2m')
                    tmpGrid = self.inst.pupilGrid.copy().scale(1/grid_diam)
                    aper *= self.inst._asym_mask(tmpGrid)
            else:
                aper = self.inst.aperture
        M2C, C2M = psi_utils.makeModalBasis(self.inst.pupilGrid,
                                            nbOfModes,
                                            nmode_shift,
                                            reortho=reortho,
                                            aperture=aper,
                                            basis_name='zern')
        
        # self.aperture_modal_basis = aper
        # self.M2C_basis = basis
        self.M2C = M2C
        self.C2M = C2M

    def buildModel(self, tag_name,
                   regenerate=False, retrain=False, auto_train=False,
                   model_path=None, nb_samples=None, nb_modes=None):
        '''
        Prepare the CNN framework.
        Generate data if necessary
        Train if necessary

        2024-03-28
            if want to train w/o adding noise during training, set the inst config noise to 0.

        TODO provide the cfg filenames instead of relying on the default set files

        PARAMETERS

        tag_name    :   string
            if you don't want to create a new dataset, you should provide a tag_name that refers to an existing
            dataset <data_path + ds_${tag_name}.h5  and have <regenerate=False>
        regenerate  :   bool
            Force data generation and override existing dataset
        retrain     :   bool
            Force training and override existing model
        auto_train  :   bool
            Automatically start training the model without asking user confirmation
        model_path  :   string
            If None (default), use the value in training_config.yml
            Otherwise, use the value given here.
        nb_samples  :   int
            If None (default), use the value in the generator_config.yml. 
            Otherwise, use the value given here.
            N.B.: only affect the dataset (the training automatically uses the complete dataset)
            TODO pass nb_samples to trainModel to be able to select a subsample of the dataset
        nb_modes    : int
            If None (default), use the number of modes in the dataset.
            Otherwise use the value given here (<= nb modes in dataset)

        '''
        # 1. Data generation
        self.logger.info('1. Data generation')
        self.generator.setup(self.inst, self.C2M, self.cfg.params)
        if nb_samples is not None:
            self.logger.info('Setting the number of samples to {0}'.format(nb_samples))
            self.generator.config['nb_samples'] = nb_samples
        data_path = self.generator.config['dataset_path']
        data_file = data_path + '/ds_{}.h5'.format(tag_name)
        pprint.pprint(self.generator.config)
        if os.path.isfile(data_file) and regenerate==False:
            self.logger.info('Data exists. Will use the existing file {0}'.format(data_file))
            # db, attrs = rt.read_h5(data_file)
            # self.generator.config = attrs
        else:
            self.logger.info('Data h5 file does not exists.'
                             ' Generating data and saving to {0}'.format(data_file))
            phasescreen_fname = self.generator.config['phasescreen_fname']
            self.generator.genData(tag_name, store_data=True,
                                   phase_fname=phasescreen_fname)
            # data generation mess up with internal variable of inst. Reloading
            self.logger.info('Reloading instrument and optical model after data generation')
            self.inst = eval(self.cfg.params.instrument)(self.cfg.params)
            self.inst.build_optical_model()
        if self.generator.config['noisy']==False:
            self.logger.info('Resetting the instrument noise setting to {0}'.format(self.cfg.params.noise))
            self.inst.noise = self.cfg.params.noise


        # 2. Training
        self.logger.info('2. Training')
        if model_path is not None:
            self.trainer.setup(training_data_fname=data_file, model_dir=model_path)
        else:
            self.trainer.setup(training_data_fname=data_file)
        self.logger.info('Model path is {0}'.format(model_path))
        model_path = self.trainer.config['model_dir']
        model_file = model_path + '/model.pth'
        if os.path.isfile(model_file) and retrain==False:
            self.logger.info('Model exists. Will use the existing `model.pth` file')
            # TODO check that the model info correspond to the data setting // use tagname ???
        else:
            self.logger.info('Model does not exist yet. Would you like to start training the model ?')
            self.logger.info('Training config: ')
            pprint.pprint(self.trainer.config)
            if auto_train is False:
                # Ask user to continue ?
                inp = input('Training ? (q to quit)')
                if inp == 'q':
                    return 0 
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            self.trainer.config['mag'] = self.cfg.params.mag
            num_photons = self.getFluxInFocalPlane() #self.inst.num_photons
            bckg_level = self.inst.bckg_level
            self.trainer.trainModel(nbModes = nb_modes,
                                    noise=self.inst.noise, 
                                    signal=num_photons,
                                    bckg=bckg_level) 

        # 3. Inference setup
        self.init_evaluator(model_fname=model_path)
        # self.logger.info('3. Preparing for inference')
        # self.evaluator.setup(model_data_path=model_path)
        # self._wavelength = self.evaluator.data_info['wavelength'] # required for consistency

        # # 3.1 Check consistency between CNN config / data_info and the current sim config
        # if self.evaluator.data_info['wavelength'] != (self.inst.wavelength * 1e9):
        #     self.logger.warning('Wavelength not the same [{0:.0f}, {1:.0f}]nm'.\
        #                         format(self.evaluator.data_info['wavelength'],
        #                                (self.inst.wavelength*1e9)))
        # # if 'config_simulation' in self.evaluator.data_info.keys():
        # if self.evaluator.data_info['config_simulation'] is True:
        #     # Cannot be exhaustive, but checking some key parameters
        #     cnn_cfg = self.evaluator.data_info #['config_simulation']
        #     sim_cfg = vars(self.cfg.params)
        #     self._check_key_configs('det_size', cnn_cfg, sim_cfg)
        #     self._check_key_configs('inst_mode', cnn_cfg, sim_cfg)
        #     # Check photometry
        #     self._check_key_configs('mag', cnn_cfg, sim_cfg)
        #     self._check_key_configs('dit', cnn_cfg, sim_cfg)
        #     self._check_key_configs('bandwidth', cnn_cfg, sim_cfg)
        #     # Check asymmetric stop
        #     self._check_key_configs('asym_angle', cnn_cfg, sim_cfg)
        #     self._check_key_configs('asym_width', cnn_cfg, sim_cfg)
        #     self._check_key_configs('asym_mask_option', cnn_cfg, sim_cfg)
        #     # Check nb modes
        #     self._check_key_configs('nb_modes', cnn_cfg, sim_cfg)
        #     self.logger.info('All configuration checks done.')
        # else:
        #     self.logger.warn('Cannot check simulation configuration')

    def init_evaluator(self, model_fname=None):
        self.logger.info('Preparing for inference')
        self.evaluator.setup(model_data_path=model_fname+'/')
        self._wavelength = self.evaluator.data_info['wavelength'] # required for consistency

        # 3.1 Check consistency between CNN config / data_info and the current sim config
        if self.evaluator.data_info['wavelength'] != (self.inst.wavelength * 1e9):
            self.logger.warning('Wavelength not the same [{0:.0f}, {1:.0f}]nm'.\
                                format(self.evaluator.data_info['wavelength'],
                                       (self.inst.wavelength*1e9)))
        # if 'config_simulation' in self.evaluator.data_info.keys():
        if self.evaluator.data_info['config_simulation'] is True:
            # Cannot be exhaustive, but checking some key parameters
            cnn_cfg = self.evaluator.data_info #['config_simulation']
            sim_cfg = vars(self.cfg.params)
            self._check_key_configs('det_size', cnn_cfg, sim_cfg)
            self._check_key_configs('inst_mode', cnn_cfg, sim_cfg)
            # Check photometry
            self._check_key_configs('mag', cnn_cfg, sim_cfg)
            self._check_key_configs('dit', cnn_cfg, sim_cfg)
            self._check_key_configs('bandwidth', cnn_cfg, sim_cfg)
            # Check asymmetric stop
            self._check_key_configs('asym_angle', cnn_cfg, sim_cfg)
            self._check_key_configs('asym_width', cnn_cfg, sim_cfg)
            self._check_key_configs('asym_mask_option', cnn_cfg, sim_cfg)
            # Check nb modes
            self._check_key_configs('nb_modes', cnn_cfg, sim_cfg)
            self.logger.info('All configuration checks done.')
        else:
            self.logger.warn('Cannot check simulation configuration')

    def getFluxInFocalPlane(self):
        noise = np.copy(self.inst.noise)
        self.inst.noise=0

        nbOfSeconds = 0.1  # 0.1 seconds is the shortest possible with CompassSim
        science_images_buffer = self.inst.grabScienceImages(nbOfSeconds)
        science_image = science_images_buffer.mean(0)
        flux = np.sum(science_image)

        self.inst.noise = noise
        return flux

    def _check_key_configs(self, kkey, cfg1, cfg2):
        if cfg1[kkey] != cfg2[kkey]:
            a = cfg1[kkey]
            b = cfg2[kkey]
            self.logger.warning('{0} not the same [{1}, {2}]'.format(kkey, a, b))

    def next(self, leak=1, gains=[None, None], integrator=True,
             display=True, save_video=False, modal_gains=1):
        '''
        PARAMETERS
        gains   : list
            gain_I and gain_P. If not [None, None], will use the given values instead of the config ones.
        '''
        nbOfSeconds = 1/self.cfg.params.framerate
        science_images_buffer = self.inst.grabScienceImages(nbOfSeconds)
        self.science_image = science_images_buffer.mean(0)
        self.inst.synchronizeBuffers(None, None)


        self._modes = self.evaluator.infer(self.science_image[np.newaxis,:,:]).squeeze()[:self.C2M.shape[0]]

        self._wavefront = self.M2C.dot(modal_gains * self._modes) * 2 * np.pi / self._wavelength
        # _dim = self.inst.pupilGrid.shape[0]
        # self._wavefront = np.reshape(_wavefront, (_dim, _dim))
        if gains == [None, None]:
            gain_I = self.cfg.params.gain_I
            gain_P = self.cfg.params.gain_P
        else:
            gain_I = gains[0]
            gain_P = gains[1]
        self._command = - gain_I * self._wavefront
        self.inst.setNcpaCorrection(self._command,
                                    phase_prop= gain_P * \
                                        self._wavefront,
                                    integrator=integrator,
                                    leak=leak)
        self.iter +=1

        if display:
            self.show(save_video=save_video)

    def loop(self, **kwargs):
        self._ims = [] # placeholder if save_video is True, see show()
        db_logger = kwargs.get('db_logger', None)
        if 'db_logger' in kwargs.keys():
            kwargs.pop('db_logger')
        for _ in range(self.cfg.params.nb_iter):
            self.next(**kwargs)
            self.evaluateSensorEstimate(db_logger=db_logger) #, aperture=self.aperture_modal_basis*self.inst.aperture)
            self.evaluateLoss()

        if self.cfg.params.save_loop_statistics:
            self._save_loop_stats()

    def evaluateLoss(self):
        modes_meas = np.copy(self._modes)

        input_phase = self.inst.phase_ncpa + self.inst.phase_wv_buffer + self.inst.phase_ncpa_correction * self.inst.aperture
        # input_phase_no_delay = self.inst.phase_ncpa + self.inst.phase_wv_buffer + \
        #     self.inst._next_phase_ncpa_correction
        if input_phase.shape[0] != 1:
            self.logger.info('strange dimension for phase_wv_buffer ?')
        modes_truth = self._C2M.dot(input_phase[-1]) * self.inst.wavelength / (2 * np.pi) * 1e9
        # modes_truth_no_delay = self._C2M.dot(input_phase_no_delay[0]) * self.inst.wavelength / (2 * np.pi) * 1e9

        wfe = np.sqrt(np.sum((modes_truth - modes_meas)**2))
        # wfe_no_delay = np.sqrt(np.sum((modes_truth_no_delay - modes_meas)**2))


        color = Fore.LIGHTRED_EX

        self.logger.info(color +'#{0} : '.format(self.iter)+ Fore.RESET + \
                            'Loss  = {0:.2f} nm'.format(wfe))
        
        self._log_modes_truth.append(modes_truth)
        self._log_modes_meas.append(modes_meas)

    def analyzeLogLoss(self):
        modes_truth = np.array(self._log_modes_truth)
        modes_meas = np.array(self._log_modes_meas)

        std_truth=np.std(modes_truth, axis=0)
        std_meas=np.std(modes_meas, axis=0)

        error = np.std(modes_truth - modes_meas, axis=0)

        plt.figure()
        plt.subplot(121)
        plt.plot(std_truth, label='std truth')
        plt.plot(std_meas, label='std meas')
        plt.semilogy()
        plt.subplot(122)
        plt.plot(error, label='std modal error')
        plt.semilogy()


    # def _modalFilteringOnEP(self, ncpa_estimate):
    #     '''
    #         Modal projection (/filtering) on entrance pupil grid
    #     '''
    #     ncpa_modes      = self.C2M.dot(ncpa_estimate.flatten() * self.inst.aperture.flatten())
    #     # ncpa_estimate  = self.M2C_large.transformation_matrix.dot(ncpa_modes)
    #     ncpa_estimate  = self.M2C_matrix.dot(ncpa_modes)

    #     return ncpa_estimate, ncpa_modes

    # def evaluateSensorEstimate(self, verbose=True):
    #     '''
    #         Compute the rms errors made on quasi-static NCPA and on water vapour seeing.

    #         /!\ Only valid for a `CompassSimInstrument` and `DemoCompassSimInstrument`

    #         TODO make it generic to any instruments
    #     '''
    #     res_ncpa_qs = self.inst.phase_ncpa + self.inst.phase_ncpa_correction
    #     res_ncpa_all = self.inst.phase_ncpa + self.inst.phase_wv + \
    #         self.inst.phase_ncpa_correction
    #     # 2022-06-2x ...
    #     if self.iter == 0:
    #         res_static_ncpa_qs = self.inst.phase_ncpa
    #     else:
    #         # tmp_avg = np.mean(self.inst.phase_ncpa_correction[self.inst.aperture>=0.5])
    #         self._ncpa_correction_long_term += self.inst.phase_ncpa_correction #- tmp_avg)
    #         # self._ncpa_correction_long_term /= self.iter

    #         res_static_ncpa_qs = self.inst.phase_ncpa + (self._ncpa_correction_long_term / self.iter)

    #     # 2022-07-01 -- metric with the average WV over one iteration
    #     res_ncpa_all_bis = self.inst.phase_ncpa + self.inst.phase_wv_integrated + \
    #         self.inst.phase_ncpa_correction

    #     conv2nm = self.inst.wavelength / (2 * np.pi) * 1e9
    #     # rms_input_qs = np.std(self.inst.phase_ncpa[self.inst.aperture==1]) * conv2nm
    #     # rms_input_all = np.std((self.inst.phase_ncpa + \
    #     #                         self.inst.phase_wv)[self.inst.aperture==1]) * conv2nm
    #     rms_res_qs = np.std(res_ncpa_qs[self.inst.aperture>=0.5]) * conv2nm
    #     rms_res_all = np.std(res_ncpa_all[self.inst.aperture>=0.5]) * conv2nm
    #     rms_res_all_bis = np.std(res_ncpa_all_bis[self.inst.aperture>=0.5]) * conv2nm


    #     if self.cfg.params.psi_correction_mode is not 'all':
    #         tmp, _ = self._modalFilteringOnEP(res_ncpa_qs)
    #         rms_res_qs_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #         tmp, _ = self._modalFilteringOnEP(res_ncpa_all)
    #         rms_res_all_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #         tmp, _ = self._modalFilteringOnEP(res_ncpa_all_bis)
    #         rms_res_all_bis_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #     else:
    #         rms_res_qs_filt = rms_res_qs
    #         rms_res_all_filt = rms_res_all

    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv)
    #     rms_wv = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_wv_integrated)
    #     rms_wv_integrated = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     tmp, _ = self._modalFilteringOnEP(self.inst.phase_ncpa_correction)
    #     rms_corr = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     tmp, _ = self._modalFilteringOnEP(res_static_ncpa_qs)
    #     rms_res_static_NCPA_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm

    #     if verbose:
    #         self.logger.info('#{0} : Res [QS, QS+WV,  QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
    #             format(self.iter, rms_res_qs, rms_res_all, rms_res_all_bis))
    #         self.logger.info('#{0} : Res. filt. [QS, QS+WV, QS+WV b] = [{1:.0f}, {2:.0f}, {3:.0f}]'.\
    #             format(self.iter, rms_res_qs_filt, rms_res_all_filt, rms_res_all_bis_filt))
    #         self.logger.info('#{0} : input WV_f rms (last, integrated)  = ({1:.0f}, {2:.0f})'.format(self.iter, rms_wv, rms_wv_integrated))
    #         self.logger.info('#{0} : PSI correction rms = {1:.0f}'.format(self.iter, rms_corr))
    #         self.logger.info('#{0} : Long-term (static) residual rms = {1:.0f}'.format(self.iter, rms_res_static_NCPA_filt))


    #     loop_stat = [self.iter]
    #     loop_stat.append(rms_res_all_filt)
    #     loop_stat.append(rms_res_qs_filt)
    #     loop_stat.append(rms_res_all)
    #     loop_stat.append(rms_res_qs)
    #     # [01/07/2022] : added 01/07/2022
    #     loop_stat.append(rms_wv_integrated)  # input WV average over 1/psi_framertae -- on the modes
    #     loop_stat.append(rms_res_all_bis_filt)    # rms all considering the average WV and not the instantaneoius
    #     loop_stat.append(rms_res_static_NCPA_filt)  # long-term average of the correction compared to the QS part
    #     self._loop_stats.append(loop_stat)

    # def show(self):
    #     ax1 = plt.subplot(141, label='science')
    #     im1, _= imshow_norm(self.science_image, stretch=LogStretch(), ax=ax1)
    #     vmin = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 1)
    #     vmax = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 99)
    #     inter = ManualInterval(vmin, vmax)
    #     #--
    #     ax2 = plt.subplot(142, label='dist')
    #     im2, _=imshow_norm((self.inst.phase_wv + self.inst.phase_ncpa).shaped,
    #                        interval=inter, ax=ax2)
    #     #--
    #     ax3 = plt.subplot(143, label='wfs')
    #     _dim = self.inst.pupilGrid.shape[0]
    #     im3, _=imshow_norm(-self.inst.phase_ncpa_correction.reshape((_dim, _dim)) * \
    #                        self.inst.aperture.shaped,
    #                        interval=inter, ax=ax3)
    #     # im3, _=imshow_norm(self._wavefront.shaped * \
    #     #                    self.inst.aperture.shaped,
    #     #                    interval=inter, ax=ax3)
    #     #--
    #     ax4 = plt.subplot(144, label='res')
    #     im4, _=imshow_norm(self.inst.aperture.shaped *
    #                        (self.inst.phase_wv + self.inst.phase_ncpa +
    #                        self.inst.phase_ncpa_correction).shaped,
    #                        interval=inter, ax=ax4)

    #     ax1.set_axis_off()
    #     ax2.set_axis_off()
    #     ax3.set_axis_off()
    #     ax4.set_axis_off()
    #     ax2.set_title('static NCPA + WV')
    #     ax3.set_title('NCPA correction')
    #     ax4.set_title('Residuals')
    #     plt.tight_layout()
    #     plt.draw()
    #     plt.pause(0.01)
    #     # time.sleep(0.1)
    #     # self._ims.append([im1, im2, im3, im4])    
    #     # 
