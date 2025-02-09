import os
import abc
from types import SimpleNamespace
import sys
# sys.path.append('')
import hcipy
# sys.path.append('/Users/orban/Projects/METIS/4.PSI/psi_github/')
import psi.psi_utils as psi_utils
from psi.psi_utils.psi_utils import crop_img, resize_img
import numpy as np
import astropy.io.fits as fits

from .helperFunctions import LazyLogger




class GenericInstrument():
    '''
    Abstract Generic instrument

    Define all the necessary methods and properties that are used by
    ``PsiSensor``.
    '''
    def __init__(self, conf, diam=1):
        '''
        Compute the pupilGrid and the focalGrid within the HCIPy framework.
        Compute phase buffer as HCIPy.Field.
        Define the propagator from pupil to focal plane as Fraunhofer propagator
        '''
        self._size_pupil_grid = conf.npupil
        self._focal_grid_resolution = conf.det_res
        self._focal_grid_size = conf.det_size

        self.pupilGrid = hcipy.make_pupil_grid(self._size_pupil_grid, diameter=diam)
        self.focalGrid = hcipy.make_focal_grid(self._focal_grid_resolution,
                                               self._focal_grid_size,
                                               pupil_diameter=diam,
                                               reference_wavelength=1,
                                               focal_length=diam)
        # GOX 08/03/2023: adding kwarg 'focal_length=diam'
        self._prop = hcipy.FraunhoferPropagator(self.pupilGrid,
                                                self.focalGrid,
                                                focal_length=diam)


        self.phase_ncpa = hcipy.Field(0.0, self.pupilGrid)         # knowledge only in Simulation
        self.phase_wv = hcipy.Field(0.0, self.pupilGrid)           # knowledge only in Simulation
        self.phase_wv_integrated = hcipy.Field(0.0, self.pupilGrid)    # knowledge only in Simulation
        self.phase_ncpa_correction = hcipy.Field(0.0, self.pupilGrid)  # NCPA correction applied
        self.phase_ncpa_correction_integrator = hcipy.Field(0.0, self.pupilGrid)

        pass

    @property
    def optical_model(self):
        '''
        HCIPy.OpticalSystem object
            a linear path of optical elements that propagates the wavefront
            forward and backward.
        '''
        return self._optical_model

    @optical_model.setter
    def optical_model(self, model):
        '''
        HCIPy.OpticalSystem object
            a linear path of optical elements that propagate the wavefront
            forward and backward.
        '''
        self._optical_model = model

    @property
    def aperture(self):
        '''
        Entrance pupil of the instrument
        '''
        return self._aperture

    @aperture.setter
    def aperture(self, aper):
        '''
        Setting the entrance pupil of the instrument
        '''
        self._aperture = aper

    @abc.abstractmethod
    def grabWfsTelemetry(self, nbOfSeconds):
        '''
        Grab wavefront sensor telemetry and returns wavefront buffer
        '''
        pass

    @abc.abstractmethod
    def grabScienceImages(self, nbOfSeconds):
        '''
        Grab science images and returns science images buffer
        '''
        pass

    @abc.abstractmethod
    def setNcpaCorrection(self):
        '''
            Set the NPCA phase correction.
        '''
        pass

    @abc.abstractmethod
    def synchronizeBuffers(self):
        '''
        Synchronize science and wfs telemetry buffers
        '''
        pass

    @abc.abstractmethod
    def getNumberOfPhotons(self):
        '''
        Provides an estimate of the total number of photons at
        the entrance pupil plane

        (calibration task)
        '''
        pass


# TODO build an abstract simulation class
# which include: build_optical_model, _generateRealisticPsf, generic setup

class CompassSimInstrument(GenericInstrument):
    def __init__(self, conf, logger=LazyLogger('CompassInstrument')):
        super().__init__(conf)

        self.logger = logger

        if type(conf) == dict:
            conf = SimpleNamespace(**conf)

        self._setup(conf)

        start_idx = 2011
        self._zero_time_ms = start_idx
        self._current_time_ms = start_idx  # starting time with COMPASS phase_screens
        self._start_time_wfs = start_idx
        self._end_time_wfs = start_idx
        self._start_time_sci_buffer = start_idx
        self._end_time_sci_buffer = start_idx
        self._start_time_last_sci_dit = start_idx
        self._end_time_last_sci_dit = start_idx

    def _setup(self, conf):
        self.wfs_exptime = 1 / conf.ao_framerate
        self.ao_frame_decimation = conf.ao_frame_decimation
        self.sci_exptime = conf.dit

        self.nb_ao_per_sci = conf.nb_ao_frames_per_science
        # self.ncpa_map = conf.ncpa_map
        # self.add_water_vapour = conf.add_water_vapour

        self.wavelength = conf.wavelength
        self._prefix_rp = conf.turb_prefix_rp
        self._prefix_wf = conf.turb_prefix_wf
        self._suffix = conf.turb_suffix
        self._input_folder = conf.turb_folder

        self._inst_mode = conf.inst_mode  # which type of imaging system

        self._asym_stop = conf.asym_stop
        if self._asym_stop:
            self._asym_angle = conf.asym_angle
            self._asym_width = conf.asym_width
            self._asym_mask = conf.asym_mask
        # Aperture definition -- GOX: see also modification later in _setup to cope with slight mismatch with the NCPA maps
        self.aperture = psi_utils.make_COMPASS_aperture(conf.f_aperture,
                                                  npupil=self._size_pupil_grid,
                                                  rot90=True,
                                                  binary=True)(self.pupilGrid)
        if  self._asym_stop and self._inst_mode == 'IMG':
            self.aperture *= self._asym_mask(self.pupilGrid)

        # self.aperture = np.rot90(self.aperture)
        if self._inst_mode == 'CVC' or self._inst_mode == 'RAVC':
            self._vc_charge = conf.vc_charge
            self._vc_vector = conf.vc_vector
            # Lyot stop mask definition ...
            self.lyot_stop_mask = psi_utils.make_COMPASS_aperture(conf.f_lyot_stop,
                                                            npupil=self._size_pupil_grid,
                                                            rot90=True)(self.pupilGrid)
            # self.lyot_stop_mask = np.rot90(self.lyot_stop_mask)
        if self._inst_mode == 'RAVC' : #or self._inst_mode == 'APP':
            self.pupil_apodizer = psi_utils.make_COMPASS_aperture(conf.f_apodizer,
                                                            npupil=self._size_pupil_grid,
                                                            rot90=True)(self.pupilGrid)

        # if self._inst_mode != 'IMG' and self._asym_stop:
        #     spider_gen = hcipy.make_spider_infinite((0,0),
        #                                             self._asym_angle,
        #                                             self._asym_width)
        #     asym_arm = spider_gen(self.pupilGrid)
        #     self.lyot_stop_mask *= asym_arm

        self.noise = conf.noise
        # if self.noise == 1:
        #     pass
        if self.noise == 2:
            self.bckg_level = conf.num_photons_bkg
        self.num_photons = conf.num_photons
        self.bandwidth = conf.bandwidth

        # by default include residual turbulence phase screens
        self.include_residual_turbulence = True
        self.phase_residual = 0
        # self.phase_residual = hcipy.Field(0.0, self.pupilGrid).shaped

        self.ncpa_dynamic = conf.ncpa_dynamic
        if self.ncpa_dynamic:
            self.ncpa_sampling = conf.ncpa_sampling
        self._input_folder_ncpa = conf.ncpa_folder
        self._prefix_ncpa = conf.ncpa_prefix
        self.ncpa_scaling = conf.ncpa_scaling
        self._initialize_dynamic_ncpa()
        # --- Customizing the entrance aperture due to some small mismatch in the NCPA definition ---
        # nb: see the 'hardcoded' +6 pixels -- same as in psi_utils.loadNCPA
        self.logger.warn('Customizing entrance aperture as a function of NCPA map definition')
        ncpa_file = self._prefix_ncpa + str(self._ncpa_index) + '.fits'
        mask_pupil = fits.getdata(self._input_folder_ncpa + ncpa_file)
        mask_pupil[mask_pupil != 0 ]=1
        # mask_pupil = resize_img(mask_pupil, self._size_pupil_grid)
        # mask_pupil= np.rot90(mask_pupil)
        size_ = self._size_pupil_grid
        mask_pupil = psi_utils.psi_utils.process_screen(mask_pupil, size_+6,
                                                  self.aperture, rotate=True, ncpa_=True)
        mask_pupil = psi_utils.psi_utils.crop_img(mask_pupil, (size_, size_))


        mask_pupil = np.ravel(mask_pupil)
        self.aperture = self.aperture * mask_pupil
        #-------------#

        self.include_water_vapour = conf.wv
        if self.include_water_vapour:
            self.wv_folder = conf.wv_folder
            self.wv_cubename = conf.wv_cubename
            self.wv_sampling = conf.wv_sampling
            self.wv_scaling = conf.wv_scaling
            self._initialize_water_vapour()
        else:
            pass
            # self.phase_wv = 0  # already initialilze in GenericInstrument
        # self.aperture = conf.aperture

        # self.phase_ncpa_correction = 0
        # ....

        # self.start_time = 2011
        # COMPASS units are µm; HCIPy needs rad
        self.conv2rad = 1e3 * (2 * np.pi / self.wavelength * 1e-9)

        # HEEPS cube are in meters:

        self.toto_scaling = 1

    def build_optical_model(self):
        '''
            Building an optical model in HCIPy depending on the instrument mode selected.

            Instrument modes are
                - CVC :  Classical Vector Coronagraph
                - ELT : normal imaging (no coronagraph, no Lyot stop)
                - RAVC : Ring-Apodized Vector Coronagraph
                - APP : Apodized Phase Plate coronagraph

        '''
        if self._inst_mode == 'CVC':
            self.logger.info('Building a Classical Vortex Coronagraph optical model in HCIPy')

            assert self._vc_charge == 2 or self._vc_charge == 4

            if self._vc_vector:
                self._vvc_element = hcipy.VectorVortexCoronagraph(self._vc_charge)
            else:
                self._vvc_element = hcipy.VortexCoronagraph(self.pupilGrid, self._vc_charge)

            self._lyot_stop_element = hcipy.Apodizer(self.lyot_stop_mask)

            self.optical_model = hcipy.OpticalSystem([self._vvc_element,
                                                      self._lyot_stop_element,
                                                      self._prop])
        # elif self._inst_mode == 'ELT' or self._inst_mode == 'IMG':
        elif self._inst_mode == 'IMG':
            self.logger.info('Building a simple imager in HCIPy')
            self.optical_model = hcipy.OpticalSystem([self._prop])

        elif self._inst_mode == 'RAVC':

            self.logger.info('Building a Ring-Apodizer Vortex Coronagraph optical model in HCIPy')

            self._ring_apodizer = hcipy.Apodizer(self.pupil_apodizer)

            assert self._vc_charge == 2 or self._vc_charge == 4

            if self._vc_vector:
                self._vvc_element = hcipy.VectorVortexCoronagraph(self._vc_charge)
            else:
                self._vvc_element = hcipy.VortexCoronagraph(self.pupilGrid, self._vc_charge)

            self._lyot_stop_element = hcipy.Apodizer(self.lyot_stop_mask)

            self.optical_model = hcipy.OpticalSystem([self._ring_apodizer,
                                                      self._vvc_element,
                                                      self._lyot_stop_element,
                                                      self._prop])

        elif self._inst_mode == 'APP':
            self.logger.warning('APP not supported')

        # # lyot_stop_mask = hcipy.make_obstructed_circular_aperture(0.98, 0.3)(pupil_grid)
        # # lyot_stop_mask = hp.evaluate_supersampled(hp.circular_aperture(0.95), pupil_grid, 4)
        # lyot_stop_mask = hp.circular_aperture(0.95)
        # lyot_stop = hcipy.Apodizer(lyot_stop_mask)

        else:
            self.logger.error('Mode {0} is not supported'.format(self._inst_mode))



    def _initialize_dynamic_ncpa(self):
        # -- NCPA should be part of the instrument ... not here ----
        self._ncpa_index = 0
        ncpa_file = self._prefix_ncpa + str(self._ncpa_index) + '.fits'
        size_pupil_grid = int(self.pupilGrid.shape[0])
        self.phase_ncpa = psi_utils.loadNCPA(self.aperture, size_pupil_grid,
                                       file_=ncpa_file,
                                       folder_=self._input_folder_ncpa,
                                       wavelength_=self.wavelength)
        self.phase_ncpa *= self.ncpa_scaling
        # # compute min max for plot
        # ncpa_min = - np.ptp(self.phase_ncpa) / 2
        # ncpa_max = np.ptp(self.phase_ncpa) / 2

    def _update_dynamic_ncpa(self, current_time):
        '''read/compute a new NCPA map'''

        if (((current_time - self._zero_time_ms)/1e3) % self.ncpa_sampling) == 0:
            self.logger.info('Updating NCPA map')
            ncpa_file = self._prefix_ncpa+str(self._ncpa_index) + '.fits'
            size_pupil_grid = int(self.pupilGrid.shape[0])
            self.phase_ncpa = psi_utils.loadNCPA(self.aperture,
                                           size_pupil_grid,
                                           file_=ncpa_file,
                                           folder_=self._input_folder_ncpa,
                                           wavelength_=self.wavelength)
            self.phase_ncpa *= self.ncpa_scaling
            self._ncpa_index += 1

    def _initialize_water_vapour(self):
        self._wv_index = 0
        # HEEPS cubes are in meters
        self.conv2rad_wv = (2 * np.pi / self.wavelength)
        self.phase_wv_cube = fits.getdata(self.wv_folder + self.wv_cubename)
        size_pupil_grid = int(self.pupilGrid.shape[0])
        self.phase_wv  = self.conv2rad_wv * \
            psi_utils.process_screen(self.phase_wv_cube[0],
                               size_pupil_grid,
                               self.aperture, rotate=True)
        self.phase_wv *= self.wv_scaling
        self.phase_wv_integrated = self.phase_wv
        # folder_wv = '/Users/orban/Projects/METIS/4.PSI/legacy_TestArea/WaterVapour/phases/'
        # file_wv = "cube_Cbasic_20210504_600s_100ms_0piston_meters_scao_only_285_WVLonly_qacits.fits"
        # wave_vapour_cube = fits.getdata(os.path.join(folder_wv, file_wv)) * \
            # 2 * np.pi / wavelength  #* 1e3 * 1e-6
        # pass

    def _update_water_vapour(self, current_time):
        '''read/compute a new NCPA map'''

        if ((current_time - self._zero_time_ms) %  self.wv_sampling) == 0:
            # self.logger.info('Updating WV map, {0}, {1}, {2}'.format(current_time, self._zero_time_ms, self.wv_sampling))
            size_pupil_grid = int(self.pupilGrid.shape[0])
            self.phase_wv  = self.conv2rad_wv * \
                psi_utils.process_screen(self.phase_wv_cube[self._wv_index],
                                   size_pupil_grid,
                                   self.aperture, rotate=True)
            self.phase_wv *= self.wv_scaling
            self._wv_index += 1



    def grabScienceImages(self, nbOfPastSeconds, **kwargs):
        '''
            Grab a buffer of science images

            Parameters
            ------------
            nbOfPastSeconds : float
                number of seconds of science images (can be equivalent to one or several images)

            Returns
            --------
            image_buffer  : numpy ndarray
                science image buffer of dimension (nbOfSciImages, nx, ny)

        '''
        self.nbOfSciImages = int(nbOfPastSeconds / self.sci_exptime)
        assert self.nbOfSciImages <= nbOfPastSeconds / self.sci_exptime
        if not(np.isclose(self.nbOfSciImages, nbOfPastSeconds/self.sci_exptime)):
            self.logger.warn('Requested buffer duration is not an integer number of Science DIT')

        nx, ny = self.focalGrid.shape
        image_buffer = np.zeros((self.nbOfSciImages, nx, ny))

        self._start_time_sci_buffer = np.copy(self._current_time_ms)
        # re-initialize timer of single dit
        self._start_time_last_sci_dit = np.copy(self._start_time_sci_buffer)
        self._end_time_last_sci_dit = np.copy(self._start_time_sci_buffer)

        if self.include_water_vapour :
            self.phase_wv_integrated = 0
            self.nb_wv_integrated = 0
        for i in range(self.nbOfSciImages):
            image_buffer[i] = self._grabOneScienceImage(bandwidth=self.bandwidth, **kwargs)

        if self.include_water_vapour :
            self.phase_wv_integrated /= self.nb_wv_integrated

        self._end_time_sci_buffer = np.copy(self._end_time_last_sci_dit)
        return image_buffer


    def _grabOneScienceImage(self, bandwidth=0, npts=11):
        '''
            Compute a single science image: consist of several realisation of the
            residual turbulence (+ NPCA, WV, NCPA_correction)


            PARAMETERS
            bandwidth: float
                spectral bandwidth. Units are dlambda/lambda.
                0 by default = monochromatic simulation.
            npts    : int
                number of points over the spectral bandwith for polychromatic simulation.
                Only used if bandwidth is > 0

        '''
        # conversion_COMPASSToNm = 1e3
        # conv = conversion_COMPASSToNm * (2 * np.pi / self.wavelength * 1e-9)
        # conv = 2 * np.pi
        nbOfFrames = int(self.sci_exptime / (self.wfs_exptime * self.ao_frame_decimation))
        deltaTime = (self.wfs_exptime * self.ao_frame_decimation) * 1e3
        timeIdxInMs = np.arange(nbOfFrames) * deltaTime

        # self._start_time_sci = np.copy(self._current_time_ms)
        # file_indices = [str(int(self._current_time_ms + timeIdxInMs[i]))
        #                 for i in range(len(timeIdxInMs))]
        self._start_time_last_sci_dit = np.copy(self._end_time_last_sci_dit)
        file_indices = [str(int(self._start_time_last_sci_dit + timeIdxInMs[i]))
                        for i in range(len(timeIdxInMs))]

        # phase in radians
        file_wf = self._prefix_rp + '_' + file_indices[0] + self._suffix
        phase_pupil = fits.getdata(os.path.join(self._input_folder, file_wf)) * self.conv2rad

        # Remove piston
        phase_pupil = psi_utils.remove_piston(phase_pupil, self.aperture.shaped)
        # conversion to HCIPy
        residual_phase = hcipy.Field(phase_pupil.ravel(), self.pupilGrid)
        wf_post_ = hcipy.Wavefront(np.exp(1j * residual_phase) * self.aperture,
                                   self.wavelength)
        # Setting number of photons
        # wf_post_.total_power = self.num_photons
        # Propagation through the instrument
        efield_fp = self.optical_model(wf_post_)
        img_one = efield_fp.power

        ss = residual_phase.shape[0]
        total_phase_cube = np.zeros((nbOfFrames, ss))

        for i in range(len(file_indices)):
            # self._current_time_ms = self._current_time_ms + timeIdxInMs[i]

            file_wf = self._prefix_rp + '_' + file_indices[i] + self._suffix

            #
            if self.include_residual_turbulence:
                self.phase_residual = fits.getdata(os.path.join(self._input_folder,
                                                           file_wf)) * self.conv2rad

                self.phase_residual = psi_utils.remove_piston(self.phase_residual, self.aperture.shaped)
                self.phase_residual *= self.toto_scaling

            # Update water vapour phase
            if self.include_water_vapour :
                # TODO timeIdxInMs needs to be a global variable: if timeIdxInMs is only a single number, 
                #   the update would happen systematically without taking wv_sampling into account
                self._update_water_vapour(self._start_time_last_sci_dit + timeIdxInMs[i])
                self.phase_wv_integrated += self.phase_wv
                self.nb_wv_integrated += 1

            # Update NCPA phase
            if self.ncpa_dynamic :
                self._update_dynamic_ncpa(self._start_time_last_sci_dit + timeIdxInMs[i])

            # Get current NCPA correction

            total_phase_cube[i] = self.phase_residual.ravel() + \
                self.phase_wv + self.phase_ncpa + self.phase_ncpa_correction

        # Forward propagation and calculation of the image for a sequence of phases
        total_phase_cube = hcipy.Field(total_phase_cube, self.pupilGrid)
        # wf_post_ = hcipy.Wavefront(np.exp(1j * total_phase_cube) * self.aperture, 1)

        # TODO  GOX (merge 08/06/2023): test mono- and poly- sequential propagation
        def _propagate_single(total_phase_cube, wlen=1):
            wf_post_0 = hcipy.Wavefront(np.exp(1j * total_phase_cube[0] / wlen) * self.aperture, wlen)
            wf_post_0.total_power = self.num_photons
            nx, ny = img_one.shaped.shape
            self._image_cube = np.zeros((nbOfFrames, nx, ny))
            self._image_cube[0] = self.optical_model(wf_post_0).power.shaped
            if nbOfFrames>1:
                for i in range(1, total_phase_cube.shape[0]):
                    wf_post_ = hcipy.Wavefront(np.exp(1j * total_phase_cube[i] / wlen) * self.aperture, wlen)
                    wf_post_.total_power = self.num_photons
                    self._image_cube[i] = self.optical_model(wf_post_).power.shaped   
            return self._image_cube
        
        if bandwidth == 0:
            self._image_cube = _propagate_single(total_phase_cube)
#             wf_post_ = hcipy.Wavefront(np.exp(1j * total_phase_cube) * self.aperture)

#             # Setting number of photons
#             # ToDo
#             wf_post_.total_power = self.num_photons * nbOfFrames
#             # Propagation through the instrument
#             # TODO: expose 'prop' and 'coro'

#             self._image_cube = self.optical_model(wf_post_).power.shaped
        else:
            assert bandwidth > 0 
            tmp_focal = 0
            for wlen in np.linspace(1 - bandwidth / 2., 1 + bandwidth / 2., npts):
                # the phase aberration needs also to be scaled to preserve the same physical OPD
#                 wf_post_ = hcipy.Wavefront(np.exp(1j * total_phase_cube / wlen) * self.aperture, wlen)
#                 wf_post_.total_power = self.num_photons * nbOfFrames
#                 tmp_focal += self.optical_model(wf_post_).power.shaped
                  tmp_focal += _propagate_single(total_phase_cube, wlen)
            self._image_cube = tmp_focal / npts

        # if vvc:
        # 	image_cube[i] = prop(coro(wf_post_)).power.shaped
        # else:
        # 	image_cube[i] = prop((wf_post_)).power.shaped
        assert len(self._image_cube.shape) == 3
        image = self._image_cube.mean(0)
        # Photometry -- TBC
        if self.noise == 0:
            noisy_image = image
        elif self.noise == 1:
            noisy_image = hcipy.large_poisson(image)
        elif self.noise == 2:
            background_noise = hcipy.large_poisson(self.bckg_level + image*0) - \
                self.bckg_level
            noisy_image = hcipy.large_poisson(image) + background_noise
        # +	np.random.poisson(nb_photons, image.shape)

        self._end_time_last_sci_dit = self._start_time_last_sci_dit + timeIdxInMs[-1] + deltaTime

        return noisy_image


    def grabWfsTelemetry(self, nbOfPastSeconds):
        '''
        Grab a buffer of WFS telemetry

        Parameters
        ------------
        nbOfPastSeconds : float
            number of seconds of science images (can be equivalent to one or several images)

        Returns
        --------
        phase_cube  : numpy ndarray
            phase cube in units of radian
        '''
        # self._compass_start_time=2011 # COMPASS 0 indexing in msec

        # conversion_COMPASSToNm = 1e3
        # conv = conversion_COMPASSToNm * (2 * np.pi / self.wavelength * 1e-9)
        nbOfFrames = int(nbOfPastSeconds / (self.wfs_exptime * self.ao_frame_decimation))
        deltaTime = (self.wfs_exptime * self.ao_frame_decimation) * 1e3
        timeIdxInMs = np.arange(nbOfFrames) * deltaTime


        self._start_time_wfs = np.copy(self._current_time_ms)
        file_indices = [str(int(self._current_time_ms + timeIdxInMs[i]))
                        for i in range(len(timeIdxInMs))]

        fname = self._prefix_wf + '_' + file_indices[0] + self._suffix
        phase_pupil = fits.getdata(os.path.join(self._input_folder, fname)) *\
            self.conv2rad

        phase_cube = np.zeros((nbOfFrames, phase_pupil.shape[0], phase_pupil.shape[1]))

        for i in range(len(file_indices)):
            # self._current_time_ms = self._current_time_ms + timeIdxInMs[i]
            file_wf = self._prefix_wf + '_' + file_indices[i] + self._suffix

            # read file
            phase = fits.getdata(os.path.join(self._input_folder, file_wf)) *\
                self.conv2rad
            # remove piston
            phase = psi_utils.remove_piston(phase, self.aperture.shaped)
            phase *= self.toto_scaling

            phase_cube[i] = np.copy(phase)

        self._end_time_wfs = self._current_time_ms + timeIdxInMs[-1] + deltaTime
        return phase_cube

    def setNcpaCorrection(self, phase_int, phase_prop=0, leak=1, integrator=True):
        '''
            Apply NCPA correction.
            Allows PI control

            Parameters
            ----------
            phase_int : numpy ndarray
                phase correction to be applied as a residual term to an integrator
            phase_prop : numpy ndarray
                proportional term to add. default is 0
            leak  : float
                leak applied to the integrator part
            integrator : bool
                True: apply leaky PI controller. False: apply `phase_int` as absolute phase correction
        '''
        if integrator:
          self.phase_ncpa_correction_integrator = leak * self.phase_ncpa_correction_integrator + phase_int
          self.phase_ncpa_correction = phase_prop + self.phase_ncpa_correction_integrator
        else:
          self.phase_ncpa_correction = phase


    def synchronizeBuffers(self, wfs_telemetry_buffer, sci_image_buffer):
        '''
            Synchronize science and wfs telemetry buffers

            Parameters
            ----------
            wfs_telemetry_buffer : numpy ndarray
                WFS telemetry buffer as returned by 'grabWfsTelemetry'
            sci_image_buffer : numpy ndarray
                Science image buffer as returned by 'grabScienceImages'


            Note:
                wfs_telemetry_buffer & sci_image_buffer are actually not used here.
                To be realistic, one could correlate the tip-tilt in both to sync them.

            Returns
            -----------
            telemetry_indexing : list
                list of start and stop index in the wfs telemetry buffer for the successive science image
        '''
        if self._start_time_wfs != self._start_time_sci_buffer:
            self.logger.warn('Start buffers not sync')
            self.logger.debug('Start WFS buffer is {0}'.format(self._start_time_wfs))
            self.logger.debug('Start SCI buffer is {0}'.format(self._start_time_sci_buffer))
            # return 0
        if self._end_time_wfs != self._end_time_sci_buffer:
            self.logger.warn('End buffers not sync')
            self.logger.debug('End WFS buffer is {0}'.format(self._end_time_wfs))
            self.logger.debug('End SCI buffer is {0}'.format(self._end_time_sci_buffer))
            # return 0

        self._current_time_ms = np.copy(self._end_time_wfs)

        # For each science image, calculate a start and stop index for
        #   the wfs telemetry buffer
        telemetry_indexing = [(i * self.nb_ao_per_sci, (i+1) * self.nb_ao_per_sci)
                           for i in range(self.nbOfSciImages)]

        return telemetry_indexing

    def getNumberOfPhotons(self):
        '''
            Returns
            --------
            Number of photons for a single science exposure
        '''
        return self.num_photons

class DemoCompassSimInstrument(CompassSimInstrument):

    def __init__(self, conf, logger=LazyLogger('CompassInstrument')):
        super().__init__(conf)

    def _grabOneScienceImage(self):
        '''
            Compute a single science image: consist of several realisation of the
            residual turbulence (+ NPCA, WV, NCPA_correction)
        '''
        # conversion_COMPASSToNm = 1e3
        # conv = conversion_COMPASSToNm * (2 * np.pi / self.wavelength * 1e-9)
        # conv = 2 * np.pi
        nbOfFrames = int(self.sci_exptime / (self.wfs_exptime * self.ao_frame_decimation))
        deltaTime = (self.wfs_exptime * self.ao_frame_decimation) * 1e3
        timeIdxInMs = np.arange(nbOfFrames) * deltaTime

        # self._start_time_sci = np.copy(self._current_time_ms)
        # file_indices = [str(int(self._current_time_ms + timeIdxInMs[i]))
        #                 for i in range(len(timeIdxInMs))]
        self._start_time_last_sci_dit= np.copy(self._end_time_last_sci_dit)
        file_indices = [str(int(self._start_time_last_sci_dit + timeIdxInMs[i]))
                        for i in range(len(timeIdxInMs))]

        # phase in radians
        file_wf = self._prefix_rp + '_'  + self._suffix
        phase_pupil = fits.getdata(os.path.join(self._input_folder, file_wf)) * self.conv2rad

        # Remove piston
        phase_pupil = psi_utils.remove_piston(phase_pupil, self.aperture.shaped)
        # conversion to HCIPy
        residual_phase = hcipy.Field(phase_pupil.ravel(), self.pupilGrid)
        # TODO remove the self.wavelength (and make sure does not affect the result -- should not of course)
        wf_post_ = hcipy.Wavefront(np.exp(1j * residual_phase) * self.aperture,
                                   self.wavelength)
        # Setting number of photons
        # wf_post_.total_power = self.num_photons
        # Propagation through the instrument
        efield_fp = self.optical_model(wf_post_)
        img_one = efield_fp.power

        nx, ny = img_one.shaped.shape
        image_cube = np.zeros((nbOfFrames, nx, ny))
        ss = residual_phase.shape[0]
        total_phase_cube = np.zeros((nbOfFrames, ss))

        for i in range(len(file_indices)):
            # self._current_time_ms = self._current_time_ms + timeIdxInMs[i]

            # file_wf = self._prefix_rp + '_' + file_indices[i] + self._suffix

            #
            if self.include_residual_turbulence:
                self.phase_residual = phase_pupil


            # Get current NCPA correction

            total_phase_cube[i] = self.phase_residual.ravel() + \
                self.phase_wv + self.phase_ncpa + self.phase_ncpa_correction

        # Forward propagation and calculation of the image for a sequence of phases
        total_phase_cube = hcipy.Field(total_phase_cube, self.pupilGrid)
        # wf_post_ = hcipy.Wavefront(np.exp(1j * total_phase_cube) * self.aperture, 1)
        wf_post_ = hcipy.Wavefront(np.exp(1j * total_phase_cube) * self.aperture)

        # TODO change to sequential propagation instead of as a tensor ! 
        #   (see CompassSim class)
        # Setting number of photons
        wf_post_.total_power = self.num_photons * nbOfFrames
        # Propagation through the instrument
        self._image_cube = self.optical_model(wf_post_).power.shaped


        assert len(self._image_cube.shape) == 3
        image = self._image_cube.mean(0)
        # Photometry -- TBC
        if self.noise == 0:
            noisy_image = image
        elif self.noise == 1:
            noisy_image = hcipy.large_poisson(image)
        elif self.noise == 2:
            background_noise = hcipy.large_poisson(self.bckg_level + image*0) - \
                self.bckg_level
            noisy_image = hcipy.large_poisson(image) + background_noise
        # +	np.random.poisson(nb_photons, image.shape)

        self._end_time_last_sci_dit = self._start_time_last_sci_dit + timeIdxInMs[-1] + deltaTime

        return noisy_image

    def grabWfsTelemetry(self, nbOfPastSeconds):
        '''
        Grab a buffer of WFS telemetry

        Parameters
        ------------
        nbOfPastSeconds : float
            number of seconds of science images (can be equivalent to one or several images)

        Returns
        --------
        phase_cube  : numpy ndarray
            phase cube in units of radian
        '''
        # self._compass_start_time=2011 # COMPASS 0 indexing in msec

        # conversion_COMPASSToNm = 1e3
        # conv = conversion_COMPASSToNm * (2 * np.pi / self.wavelength * 1e-9)
        nbOfFrames = int(nbOfPastSeconds / (self.wfs_exptime * self.ao_frame_decimation))
        deltaTime = (self.wfs_exptime * self.ao_frame_decimation) * 1e3
        timeIdxInMs = np.arange(nbOfFrames) * deltaTime


        self._start_time_wfs = np.copy(self._current_time_ms)
        file_indices = [str(int(self._current_time_ms + timeIdxInMs[i]))
                        for i in range(len(timeIdxInMs))]

        fname = self._prefix_wf + '_'  + self._suffix
        phase_pupil = fits.getdata(os.path.join(self._input_folder, fname)) *\
            self.conv2rad

        phase_cube = np.zeros((nbOfFrames, phase_pupil.shape[0], phase_pupil.shape[1]))

        for i in range(len(file_indices)):
            # self._current_time_ms = self._current_time_ms + timeIdxInMs[i]
            # file_wf = self._prefix_wf + '_' + file_indices[i] + self._suffix

            # # read file
            # phase = fits.getdata(os.path.join(self._input_folder, file_wf)) *\
            #     self.conv2rad
            # # remove piston
            # phase = psi_utils.remove_piston(phase, self.aperture.shaped)
            # phase *= self.toto_scaling

            phase_cube[i] = np.copy(phase_pupil)

        self._end_time_wfs = self._current_time_ms + timeIdxInMs[-1] + deltaTime
        return phase_cube

class HcipySimInstrument(GenericInstrument):
    '''
        TODO check instrument type in conf.params corresponds to the class name ... ?
    '''
    def __init__(self, conf, logger=LazyLogger('HcipySim')):
        '''
            Physical dimension is requested here for the aperture definition,
            because of physical simulation of the residual atmospheric turbulence.
        '''
        self.norm_telDiam = True

        super().__init__(conf, conf.tel_diam)

        self.logger = logger

        if type(conf) == dict:
            conf = SimpleNamespace(**conf)

        self._setup(conf)
        
        start_time = 0
        self._current_time_ms = start_time
        self._start_time_wfs = start_time
        self._end_time_wfs = start_time

        self._start_time_sci_buffer = start_time
        self._end_time_sci_buffer = start_time
        self._start_time_last_sci_dit = start_time
        self._end_time_last_sci_dit = start_time
        self._buffer_time_index = [start_time]


    def _setup(self, conf):
        '''
        TODO constructing asymmetric aperture should be 'generic' 
        '''
        self.diam = conf.tel_diam
        if conf.pupil == 'ELT':
            self.aperture = hcipy.aperture.make_elt_aperture(normalized=False)(self.pupilGrid)
        elif conf.pupil == 'ERIS':
            self.aperture = hcipy.aperture.make_vlt_aperture(normalized=False,
                                                             telescope='ut4',
                                                             with_spiders=True,
                                                             with_M3_cover=True)(self.pupilGrid)
        elif conf.pupil == 'CIRC':
            self.aperture = hcipy.aperture.make_circular_aperture(self.diam)(self.pupilGrid)

        self.nb_ao_per_sci = conf.nb_ao_frames_per_science
        self.decimation = conf.ao_frame_decimation #10

        self.wfs_exptime = 1 / conf.ao_framerate
        # self.ao_frame_decimation = conf.ao_frame_decimation
        self.sci_exptime = conf.dit


        self.wavelength = conf.wavelength
        self._inst_mode = conf.inst_mode  # which type of imaging system

        self._asym_stop = conf.asym_stop
        if self._asym_stop:
            self._asym_angle = conf.asym_angle
            self._asym_width = conf.asym_width
            self._asym_mask = conf.asym_mask

        if  self._asym_stop and self._inst_mode == 'IMG':
            self.aperture *= self._asym_mask(self.pupilGrid)

        if self._inst_mode == 'CVC' or self._inst_mode == 'RAVC':
            self.logger.warning('CVC / RAVC not implemented')
            # self._vc_charge = conf.vc_charge
            # self._vc_vector = conf.vc_vector
            # # Lyot stop mask definition ...
            # self.lyot_stop_mask = psi_utils.make_COMPASS_aperture(conf.f_lyot_stop,
            #                                                 npupil=self._size_pupil_grid,
            #                                                 rot90=True)(self.pupilGrid)
            # self.lyot_stop_mask = np.rot90(self.lyot_stop_mask)
        if self._inst_mode == 'RAVC' : #or self._inst_mode == 'APP':
            self.logger.warning('CVC / RAVC not implemented')
            # self.pupil_apodizer = psi_utils.make_COMPASS_aperture(conf.f_apodizer,
            #                                                 npupil=self._size_pupil_grid,
            #                                                 rot90=True)(self.pupilGrid)

        # if self._inst_mode != 'ELT' and self._asym_stop:
        #     spider_gen = hcipy.make_spider_infinite((0,0),
        #                                             self._asym_angle,
        #                                             self._asym_width)
        #     asym_arm = spider_gen(self.pupilGrid)
        #     self.lyot_stop_mask *= asym_arm

        self.noise = conf.noise
        if self.noise == 2:
            self.bckg_level = conf.num_photons_bkg
        self.num_photons = conf.num_photons
        self.bandwidth = conf.bandwidth

        self._setup_modal_basis()
        
        # by default include residual turbulence phase screens
        self.phase_residual = hcipy.Field(0.0, self.pupilGrid) 
        self.include_residual_turbulence = False
        if conf.residual_turbulence:
            self.include_residual_turbulence = True
            self._setup_modal_ao()

        self.ncpa_scaling = conf.ncpa_scaling
        self._initialize_ncpa(conf.ncpa_coefficients)
        # self.ncpa_dynamic = conf.ncpa_dynamic
        # if self.ncpa_dynamic:
        #     self.ncpa_sampling = conf.ncpa_sampling
        self.include_water_vapour = False
        pass

    def _setup_modal_basis(self):
        pupil_grid = self.pupilGrid
        #if self._inst_mode == 'ELT':
        if self.tel_diam > 20:
            self._nmodes = 1500
        else:
            self._nmodes = 500

        self.logger.info('Generating modal basis')
        zernike_modes = hcipy.make_zernike_basis(self._nmodes + 1, self.diam, pupil_grid, radial_cutoff=False)
        # mask = self.aperture 
        # mask[self.aperture >=0.7] = 1
        # mask[self.aperture<0.7] = 0
        mask = self.aperture    # assume that self.aperture is binary
        zernike_modes = psi_utils.reorthonormalize(zernike_modes, mask)

        self.ao_modes = hcipy.ModeBasis([mode * self.aperture for mode in zernike_modes]) # could also do some normalization


    def _setup_modal_ao(self):
        r0 = 0.15           # [m]
        L0 = 25           # [m]  
        wind_velocity = 16 # [m/s]   # [GOX] : very fast wind speed for fast PSI simulation 
        lag = 3          # number of time step between measurement and correction
        self.sampling_time = 1e-3 # [s] time step or sampling time 
        self._decimals = int(- np.log10(self.sampling_time)) + 1 # Precision use for list indexing, see self._buffer_time_index.index
        # nmodes = 500
        pupil_grid = self.pupilGrid

        # Define some AO behaviour
        # ao_modes = make_gaussian_influence_functions(pupil_grid, ao_actuators, 1.0 / ao_actuators)	# Create an object containing all the available DM pistons, 1.0 to
        # ao_modes = ModeBasis([mode * aperture for mode in ao_modes])
        # transformation_matrix = ao_modes.transformation_matrix
        # reconstruction_matrix = inverse_tikhonov(transformation_matrix, reconstruction_normalisation)
        

        # Instantiate an atmosphere class. The idea is then to set the electric field as to that from the COMPASS residuals
        self.logger.info('Generating ModalAdaptiveOpticsLayer')
        self.layer = hcipy.ModalAdaptiveOpticsLayer(hcipy.InfiniteAtmosphericLayer(pupil_grid,
                                                        hcipy.Cn_squared_from_fried_parameter(r0),
                                                        L0,
                                                        wind_velocity,
                                                        use_interpolation=True), 
                                        self.ao_modes, lag)
    
    def build_optical_model(self):
        '''
            Building an optical model in HCIPy depending on the instrument mode selected.

            Instrument modes are
                - CVC :  Classical Vector Coronagraph
                - ELT : normal imaging (no coronagraph, no Lyot stop)
                - RAVC : Ring-Apodized Vector Coronagraph
                - APP : Apodized Phase Plate coronagraph

        '''
        if self._inst_mode == 'CVC':
            self.logger.info('Building a Classical Vortex Coronagraph optical model in HCIPy')

            assert self._vc_charge == 2 or self._vc_charge == 4

            if self._vc_vector:
                self._vvc_element = hcipy.VectorVortexCoronagraph(self._vc_charge)
            else:
                self._vvc_element = hcipy.VortexCoronagraph(self.pupilGrid, self._vc_charge)

            self._lyot_stop_element = hcipy.Apodizer(self.lyot_stop_mask)

            self.optical_model = hcipy.OpticalSystem([self._vvc_element,
                                                      self._lyot_stop_element,
                                                      self._prop])
        elif self._inst_mode == 'IMG':
            self.logger.info('Building a simple imager in HCIPy')
            self.optical_model = hcipy.OpticalSystem([self._prop])

        elif self._inst_mode == 'RAVC':

            self.logger.info('Building a Ring-Apodizer Vortex Coronagraph optical model in HCIPy')

            self._ring_apodizer = hcipy.Apodizer(self.pupil_apodizer)

            assert self._vc_charge == 2 or self._vc_charge == 4

            if self._vc_vector:
                self._vvc_element = hcipy.VectorVortexCoronagraph(self._vc_charge)
            else:
                self._vvc_element = hcipy.VortexCoronagraph(self.pupilGrid, self._vc_charge)

            self._lyot_stop_element = hcipy.Apodizer(self.lyot_stop_mask)

            self.optical_model = hcipy.OpticalSystem([self._ring_apodizer,
                                                      self._vvc_element,
                                                      self._lyot_stop_element,
                                                      self._prop])

        elif self._inst_mode == 'APP':
            self.logger.warning('APP not supported')
        else:
            self.logger.error('Mode {0} is not supported'.format(self._inst_mode))


    def _initialize_ncpa(self, coeffs, modal_basis=None):
        '''

        PARAMETERS
        ----------
        coeffs  : 1d array
            vector of modal coefficients. Len(coeffs) should be <= # of modes in the modal basis
        moda_basis  : hcipy.ModeBasis
            (optional). If not set, uses the self.ao_modes
        '''
        if modal_basis is None:
            self.logger.info('Using AO modal basis to generate NCPA map')
            # generate a default phase map
            self.phase_ncpa = self.ao_modes[:len(coeffs)].linear_combination(coeffs)
        else:
            self.phase_ncpa = modal_basis[:len(coeffs)].linear_combination(coeffs)

    def set_ncpa(self, phase_ncpa):
            assert phase_ncpa.ndim == 1, 'phase ncpa should be a flatten (Field) array'
            self.phase_ncpa = phase_ncpa

    def setNcpaCorrection(self, phase, integrator=True, leak=1):
        if integrator:
            self.phase_ncpa_correction = leak * self.phase_ncpa_correction + phase
        else:
            self.phase_ncpa_correction = phase

    def _initialize_water_vapour(self):
        raise NotImplemented

    def _generateRealisticPsf(self, phase, total_num_photons,
                              bandwidth=0, npts=11, noise=0):
        '''
            PARAMETERS
            ------------
            phase : 2D array
                phase aberrations of dimensions (nbOfFrames x pupil_size). 
                NbOfFrames is the number of realization per one readout

            total_num_photons : int
                total number of photons = num_photons_per_frames x nbOfFrames
                
            bandwidth    : float
                normalized chromatic bandwidth. 0 for monochromatic simulation
                should be <= 1
            
            npts        : int
                number of points for polychromatic simulation. Ignored if bandwidth==0
            
            noise   : int
                option for photometry. 
                0 : no noise
                1 : Poisson noise
                2 : Poisson noise + background noise
            
            RETURNS
            --------
            noisy_image : hcipy.Field
                final image 
        '''
        ## Polychromatic
        if bandwidth == 0:
            wf_post_ = hcipy.Wavefront(np.exp(1j * phase) * self.aperture)

            # Setting number of photons
            wf_post_.total_power = total_num_photons   #self.num_photons * nbOfFrames
            self._image_cube = self.optical_model(wf_post_).power.shaped
        else:
            assert bandwidth > 0 
            tmp_focal = 0
            for wlen in np.linspace(1 - bandwidth / 2., 1 + bandwidth / 2., npts):
                # the phase aberration needs also to be scaled to preserve the same physical OPD
                wf_post_ = hcipy.Wavefront(np.exp(1j * phase / wlen) * self.aperture, wlen)
                wf_post_.total_power = total_num_photons
                tmp_focal += self.optical_model(wf_post_).power.shaped
            self._image_cube = tmp_focal / npts

        assert len(self._image_cube.shape) == 3
        image = self._image_cube.mean(0)
        ## Photometry
        if self.noise == 0:
            noisy_image = image
        elif self.noise == 1:
            noisy_image = hcipy.large_poisson(image)
        elif self.noise == 2:
            background_noise = hcipy.large_poisson(self.bckg_level + image*0) - \
                self.bckg_level
            noisy_image = hcipy.large_poisson(image) + background_noise

        return noisy_image


    def _generateWfsTelemetry(self, phase):
        ''''
           Generate wavefront sensor telemetry based on an input residual phase map

           Currently, simply project the phase screen on the finite set of ao modes
            (defined in the ModalAOLayer)

            TODO: add sensor effects like aliasing, meas. noise, etc.

            PARAMETERS
            -----------
            phase : hcipy.Field
                residual phase aberrations 
            
            RETURNS
            -----------
            wfs telemetry array (hcipy.Field)
        '''
        coeffs = self.layer.transformation_matrix_inverse.dot(phase * self.aperture)
        telemetry = self.ao_modes.linear_combination(coeffs)   
        return telemetry

    def _evolve_atmosphere(self, nbOfSeconds, reset=True):
        if reset:
            self._res_phase_buffer = []
            self._telemetry_buffer = []
            self._buffer_time_index = []
        
        # rounding to 1e-4 sec
        delta_t = self.sampling_time
        time_steps = np.round(np.arange(self._current_time_ms * 1e-3,
                               self._current_time_ms * 1e-3 + nbOfSeconds + delta_t,
                               delta_t), decimals=4)

        for time_step in time_steps:
            # Residual phase
            self.layer.evolve_until(time_step)
            # phase_screen_phase = self.layer.phase_for(1)# in radian
            phase_screen_phase = self.layer.phase_for(self.wavelength)
            telemetry = self._generateWfsTelemetry(phase_screen_phase)

            # Save time buffer separately. No decimation is applied here
            self._res_phase_buffer.append(phase_screen_phase)
            self._telemetry_buffer.append(telemetry)
            self._buffer_time_index.append(time_step)

        # self._current_time_ms = time_steps[-1] * 1e3

    # def _checkIfDataIsAvailable(self, nbOfPastSeconds):
    #     available_time_lapse = self._buffer_time_index[-1] - self._buffer_time_index[0]
    #     if available_time_lapse < nbOfPastSeconds:
    #         self.logger.warn("Past not available, evolving atmosphere")
    #         self._evolve_atmosphere(nbOfPastSeconds - available_time_lapse)

    def _checkEvolutionAtmosphere(self, nbOfPastSeconds):
        if (self._current_time_ms * 1e-3 + nbOfPastSeconds) > self.layer.layer.t:
            self.logger.warn("Past not available, evolving atmosphere")
            delta_time =  (self._current_time_ms * 1e-3 + nbOfPastSeconds) - self.layer.layer.t
            self.logger.warn("Evolving atmosphere for {0:.5f} seconds".format(delta_time))
            self._evolve_atmosphere(delta_time)

    def grabWfsTelemetry(self, nbOfPastSeconds):
        assert self.include_residual_turbulence, \
            "Can only grab wavefront sensor telemetry if simulating residual atmospheric turbulence"
        
        self._checkEvolutionAtmosphere(nbOfPastSeconds)
        # Select data in the telemetry buffer
        st = self.layer.layer.t #self._current_time_ms * 1e-3 

        
        start = self._buffer_time_index.index(np.round(st - nbOfPastSeconds + self.sampling_time,
                                                       self._decimals))
        end = self._buffer_time_index.index(np.round(st, self._decimals))

        self._start_time_wfs = self._buffer_time_index[start]
        self._end_time_wfs = self._buffer_time_index[end]
        telemetry_selected = self._telemetry_buffer[start:end:self.decimation]
        nbOfFrames = len(telemetry_selected)
        # Reshape to 3D cube
        return np.array(telemetry_selected).reshape((nbOfFrames,
                                                     self.pupilGrid.dims[0],
                                                     self.pupilGrid.dims[1]))


    def _grabOneScienceImage(self, bandwidth=0, npts=11):

        # nbOfPhaseRealization = self.sci_exptime / (self.wfs_exptime * self.ao_frame_decimation)
        #delta_t = 1e-3

        if self.include_residual_turbulence:
            nbOfFrames = int(self.sci_exptime / (self.sampling_time * self.decimation)) 
            # Handling of time and select appropriate residual phase screens
            self._start_time_last_sci_dit = np.copy(self._end_time_last_sci_dit + self.sampling_time)
            start_sci = self._buffer_time_index.index(np.round(self._start_time_last_sci_dit, self._decimals))# + self.sampling_time)
            end_sci = self._buffer_time_index.index(np.round(self._start_time_last_sci_dit +
                                                             self.sci_exptime - self.sampling_time,
                                                             self._decimals))
            self._end_time_last_sci_dit = np.copy(self._start_time_last_sci_dit + self.sci_exptime - self.sampling_time)

            res_phase_selected = self._res_phase_buffer[start_sci:end_sci:self.decimation]

        else:
            #self.logger.warn('Setting science image to a single realization')
            nbOfFrames = 1

        ss = self.pupilGrid.points.shape[0]
        total_phase_cube = np.zeros((nbOfFrames, ss))

        for i in range(nbOfFrames):  
            if self.include_residual_turbulence:
                self.phase_residual = res_phase_selected[i]

            # TODO Update water vapour phase
            # TODO Update NCPA phase

            total_phase_cube[i] = self.phase_residual + \
                self.phase_wv + self.phase_ncpa + self.phase_ncpa_correction

        total_phase_cube = hcipy.Field(total_phase_cube, self.pupilGrid)

        noisy_image = self._generateRealisticPsf(total_phase_cube,
                                                 self.num_photons*nbOfFrames,
                                                 bandwidth=bandwidth,
                                                 npts=npts,
                                                 noise=self.noise)


        return noisy_image

    def grabScienceImages(self, nbOfPastSeconds, **kwargs):
        self.nbOfSciImages = int(nbOfPastSeconds / self.sci_exptime)
        if self.include_residual_turbulence:
            self._checkEvolutionAtmosphere(nbOfPastSeconds)
            # select data in the telemetry buffer
            st = self.layer.layer.t #self._current_time_ms * 1e-3
            start_idx = self._buffer_time_index.index(np.round(st - nbOfPastSeconds + self.sampling_time, self._decimals))
            self._start_time_sci_buffer = self._buffer_time_index[start_idx]
            self._end_time_sci_buffer = st

            self._start_time_last_sci_dit = np.copy(self._start_time_sci_buffer)
            self._end_time_last_sci_dit = np.copy(self._start_time_sci_buffer - self.sampling_time)

            assert self.nbOfSciImages <= nbOfPastSeconds / self.sci_exptime
            if not(np.isclose(self.nbOfSciImages, nbOfPastSeconds/self.sci_exptime)):
                self.logger.warn('Requested buffer duration is not an integer number of Science DIT')

        if self.include_water_vapour :
            self.phase_wv_integrated = 0
            self.nb_wv_integrated = 0

        # Generating all the science images
        nx, ny = self.focalGrid.shape
        image_buffer = np.zeros((self.nbOfSciImages, nx, ny))
        for i in range(self.nbOfSciImages):
            image_buffer[i] = self._grabOneScienceImage(bandwidth=self.bandwidth,
                                                        **kwargs)

        # Various handling
        if self.include_water_vapour :
            self.phase_wv_integrated /= self.nb_wv_integrated

        if self.include_residual_turbulence:
            self._end_time_sci_buffer = np.copy(self._end_time_last_sci_dit)

        # Returns
        return image_buffer

    def synchronizeBuffers(self):
        return self.synchronizeBuffers(None, None)

    def synchronizeBuffers(self, wfs_telemetry_buffer, science_images_buffer):
        # raise NotImplementedError()
        # self._evolve_atmosphere(nbOfSeconds)
        self._current_time_ms = self.layer.layer.t * 1e3
        
        # For each science image, calculate a start and stop index for
        #   the wfs telemetry buffer
        telemetry_indexing = [(i * self.nb_ao_per_sci, (i+1) * self.nb_ao_per_sci)
                           for i in range(self.nbOfSciImages)]

        return telemetry_indexing

    def getNumberOfPhotons(self):
        '''
            Returns
            --------
            Number of photons for a single science exposure
        '''
        return self.num_photons

class ErisInterfaceOffline(GenericInstrument):
    '''
    Not implemented
    '''
    # raise NotImplementedError()
    pass


# if __name__ == '__main__':
#     from configParser import loadConfiguration
#     config_file = '/Users/orban/Projects/METIS/4.PSI/psi_github/config/config_metis_compass.py'
#     cfg = loadConfiguration(config_file)
#     inst = CompassSimInstrument(cfg.params)
#     inst.build_optical_model()
