

import abc
import os
import time
from colorama import Fore
import numpy as np
# import hcipy
# import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.visualization import imshow_norm,\
    SqrtStretch, MinMaxInterval, PercentileInterval, \
    LinearStretch, SinhStretch, LogStretch, ManualInterval
from astropy.stats import mad_std

class AbstractSensor():
    '''
    Abstract Focal Plane Wavefront Sensor class
    '''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self._ncpa_correction_long_term = 0
        self._axes = None

    # @property
    # def inst(self):
    #     '''
    #     Instrument
    #     '''
    #     return self._inst

    # @inst.setter
    # def inst(self, x):
    #     '''
    #     Setting the instrument
    #     '''
    #     self._inst = x

    # @property
    # def iter(self):
    #     '''
    #     Commands-to-modes matrix
    #     '''
    #     return self._iter
    
    @property
    def C2M(self):
        '''
        Commands-to-modes matrix
        '''
        return self._C2M

    @C2M.setter
    def C2M(self, C2M):
        '''
        Commands-to-modes matrix
        '''
        self._C2M = C2M

    @property
    def M2C(self):
        '''
        Modes-to-command matrix
        '''
        return self._M2C

    @M2C.setter
    def M2C(self, M2C):
        '''
        Setting the instrument
        '''
        self._M2C = M2C

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def next(self):
        pass

    @abc.abstractmethod
    def loop(self):
        pass

    def _modal_filtering_on_pupil(self, wavefront):
        '''
        Modal projection on the entrance pupil

        PARAMETERS
        wavefront   :   2d numpy array
        '''
        # modes = self._C2M.dot(wavefront.flatten() * self.inst.aperture.flatten())
        modes = self._C2M.dot((wavefront * self.inst.aperture).T)
        wf_filtered = self._M2C.dot(modes).T
        return wf_filtered, modes

    def evaluateSensorEstimate(self, verbose=True, static=False, db_logger=None, aperture=None):
        '''
        Compute the rms errors made on quasi-static NCPA and on water vapour seeing.

        TODO include evaluation on Lyot stop for VC cases
        '''
        # * Quasi-static residual aberrations
        res_ncpa_qs = self.inst.phase_ncpa + self.inst.phase_ncpa_correction
        # * Residual differential aberrations (all except SCAO residuals)
        # * NB: using the WV integrated phase instead of the instantaneous WV phase
        # res_ncpa_all = self.inst.phase_ncpa + self.inst.phase_wv_integrated + \
        #     self.inst.phase_ncpa_correction
        res_ncpa_all = self.inst.phase_ncpa + self.inst.phase_wv_buffer + \
            self.inst.phase_ncpa_correction        
        
        # Measurement error: not taking the delay into account
        # res_ncpa_all_no_delay = self.inst.phase_ncpa + self.inst.phase_wv_integrated + \
        #     self.inst._next_phase_ncpa_correction
        res_ncpa_all_no_delay = self.inst.phase_ncpa + self.inst.phase_wv_buffer + \
            self.inst._next_phase_ncpa_correction
        
        if aperture is None:
            aperture = self.inst.aperture
        
        if static:
            # Purely static residual aberrations (``double integration'')
            if self.iter == 0:
                res_static_ncpa_qs = self.inst.phase_ncpa
            else:
                self._ncpa_correction_long_term += self.inst.phase_ncpa_correction
                res_static_ncpa_qs = self.inst.phase_ncpa + (self._ncpa_correction_long_term / self.iter)

            tmp, _ = self._modal_filtering_on_pupil(res_static_ncpa_qs)
            rms_res_static_NCPA_filt = np.std(tmp[aperture>=0.5]) * conv2nm



        # Compute rms WFE in nm
        conv2nm = self.inst.wavelength / (2 * np.pi) * 1e9
        # rms_wv_integrated = np.std(self.inst.phase_wv_integrated[aperture>=0.5]) * conv2nm
        rms_wv = np.mean(np.std(self.inst.phase_wv_buffer,
                                     axis=1,
                                     where=aperture>=0.5)) * conv2nm
        rms_res_qs = np.std(res_ncpa_qs[aperture>=0.5]) * conv2nm
        rms_res_all = np.mean(np.std(res_ncpa_all,
                                     axis=1,
                                     where=aperture>=0.5)) * conv2nm
        rms_res_all_no_delay = np.mean(np.std(res_ncpa_all_no_delay,
                                              axis=1,
                                              where=aperture>=0.5)) * conv2nm


        # Modal filtering and recomputation of WFE
        # tmp, _ = self._modal_filtering_on_pupil(self.inst.phase_wv)
        # rms_wv_filt = np.std(tmp[self.inst.aperture>=0.5]) * conv2nm
        tmp, _ = self._modal_filtering_on_pupil(self.inst.phase_wv_buffer)
        rms_wv_filt = np.mean(np.std(tmp, axis=1, where=aperture>=0.5)) * conv2nm 
        tmp, _ = self._modal_filtering_on_pupil(self.inst.phase_ncpa_correction)
        rms_correction_filt = np.std(tmp[aperture>=0.5]) * conv2nm

        tmp, _ = self._modal_filtering_on_pupil(res_ncpa_qs)
        rms_res_qs_filt = np.std(tmp[aperture>=0.5]) * conv2nm
        tmp, _ = self._modal_filtering_on_pupil(res_ncpa_all)
        rms_res_all_filt = np.mean(np.std(tmp, axis=1, where=aperture>=0.5)) * conv2nm 
        
        tmp, _ = self._modal_filtering_on_pupil(res_ncpa_all_no_delay)
        rms_res_all_filt_no_delay = np.mean(np.std(tmp, axis=1, where=aperture>=0.5)) * conv2nm 

        if self.inst._inst_mode in ['CVC', 'RAVC']:
            pup = self.inst.lyot_stop_mask
            # lyot_rms_wv = np.std(self.inst.phase_wv_integrated[pup>=0.5]) * conv2nm
            lyot_rms_wv = np.mean(np.std(self.inst.phase_wv_buffer,
                                         axis=1,
                                         where=pup>=0.5)) * conv2nm
            lyot_rms_res_qs = np.std(res_ncpa_qs[pup>=0.5]) * conv2nm
            # lyot_rms_res_all = np.std(res_ncpa_all[pup>=0.5]) * conv2nm
            lyot_rms_res_all = np.mean(np.std(res_ncpa_all,axis=1,
                                     where=pup>=0.5)) * conv2nm

            tmp, _ = self._modal_filtering_on_pupil(self.inst.phase_wv_buffer)
            lyot_rms_wv_filt = np.mean(np.std(tmp, axis=1, where=pup>=0.5)) * conv2nm
            tmp, _ = self._modal_filtering_on_pupil(self.inst.phase_ncpa_correction)
            lyot_rms_correction_filt = np.std(tmp[pup>=0.5]) * conv2nm

            tmp, _ = self._modal_filtering_on_pupil(res_ncpa_qs)
            lyot_rms_res_qs_filt = np.std(tmp[pup>=0.5]) * conv2nm
            tmp, _ = self._modal_filtering_on_pupil(res_ncpa_all)
            # lyot_rms_res_all_filt = np.std(tmp[pup>=0.5]) * conv2nm
            lyot_rms_res_all_filt = np.mean(np.std(tmp, axis=1, where=pup>=0.5)) * conv2nm

        if verbose:
            if np.mod(self.iter, 2):
                color = Fore.LIGHTGREEN_EX
            else:
                color = Fore.LIGHTBLUE_EX
            nb_modes = self._C2M.shape[0]

            self.logger.info(color +'#{0} : '.format(self.iter)+ Fore.RESET + \
                             'Input WV [all, {0:.0f} modes]  = '.format(nb_modes) + \
                             '({0:.0f}, {1:.0f})'.\
                            format(rms_wv, rms_wv_filt))
                            #  format(rms_wv_integrated, rms_wv_integrated_filt))
            self.logger.info(color +'#{0} : '.format(self.iter)+ Fore.RESET + \
                             'Residuals [all, {0:.0f} modes] = '.format(nb_modes)+\
                             '[{0:.0f}, {1:.0f}]'.\
                             format(rms_res_all, rms_res_all_filt))
            self.logger.info(color +'#{0} : '.format(self.iter)+ Fore.RESET + \
                             'Residuals no delay [all, {0:.0f} modes] = '.format(nb_modes)+\
                             '[{0:.0f}, {1:.0f}]'.\
                             format(rms_res_all_no_delay, rms_res_all_filt_no_delay))
            # self.logger.info(color +'#{0} :'+ Fore.RESET + 'Residuals [QS, QS+WV] = [{1:.0f}, {2:.0f}]'.\
            #                  format(self.iter, rms_res_qs, rms_res_all))
            # self.logger.info(color +'#{0} :'+ Fore.RESET + ' Residuals on {1:.0f} modes [QS, QS+WV] = [{1:.0f}, {2:.0f}]'.\
            #                  format(self.iter, nb_modes, rms_res_qs_filt, rms_res_all_filt))
            if self.inst._inst_mode in ['CVC', 'RAVC']:
                self.logger.info(color +'#{0} : '.format(self.iter)+ Fore.RESET + \
                                 'Input WV Lyot [all, {0:.0f} modes]  = '.format(nb_modes)+\
                                 '({0:.0f}, {1:.0f})'.\
                                format(lyot_rms_wv, lyot_rms_wv_filt))
                                # format(lyot_rms_wv_integrated, lyot_rms_wv_integrated_filt))
                self.logger.info(color +'#{0} : '.format(self.iter)+ Fore.RESET + \
                                 'Residuals Lyot [all, {0:.0f} modes] = '.format(nb_modes)+\
                                 '[{0:.0f}, {1:.0f}]'.\
                                format(lyot_rms_res_all, lyot_rms_res_all_filt))
                
            self.logger.info(color +'#{0} :'.format(self.iter)+ Fore.RESET + ' Sensor correction rms = {0:.0f}'.\
                             format(rms_correction_filt))
            if static:
                self.logger.info(color +'#{0} :'.format(self.iter)+ Fore.RESET + ' Long-term (static) residual rms = {0:.0f}'.\
                                 format(rms_res_static_NCPA_filt))
            
        if self.iter == 1:
            loop_legend = ['it']
            loop_legend.append('wfe_all_f_avg')
            loop_legend.append('wfe_qs_f')
            loop_legend.append('wfe_all')
            loop_legend.append('wfe_qs')
            loop_legend.append('input_wv_avg')
            loop_legend.append('input_wv_f_avg')
            if self.inst._inst_mode in ['CVC', 'RAVC']:
                loop_legend.append('input_wv_avg_lyot')
                loop_legend.append('input_wv_f_avg_lyot')
                loop_legend.append('wfe_all_avg_lyot')
                loop_legend.append('wfe_all_f_avg')
            self.loop_legend = loop_legend

        loop_stat = [self.iter]
        loop_stat.append(rms_res_all_filt)
        loop_stat.append(rms_res_qs_filt)
        loop_stat.append(rms_res_all)
        loop_stat.append(rms_res_qs)
        # [01/07/2022] : added 01/07/2022
        # loop_stat.append(rms_wv_integrated)  # input WV average over 1/psi_framertae -- on the modes
        # loop_stat.append(rms_wv_integrated_filt)  # input WV average over 1/psi_framertae -- on the modes
        loop_stat.append(rms_wv)         
        loop_stat.append(rms_wv_filt)   

        # loop_stat.append(rms_res_all_bis_filt)    # rms all considering the average WV and not the instantaneoius
        # if static:
        #     loop_stat.append(rms_res_static_NCPA_filt)  # long-term average of the correction compared to the QS part
        if self.inst._inst_mode in ['CVC', 'RAVC']:
            # loop_stat.append(lyot_rms_wv_integrated)
            # loop_stat.append(lyot_rms_wv_integrated_filt)
            loop_stat.append(lyot_rms_wv)
            loop_stat.append(lyot_rms_wv_filt)
            loop_stat.append(lyot_rms_res_all)
            loop_stat.append(lyot_rms_res_all_filt)

        self._loop_stats.append(loop_stat)

        if db_logger is not None:
            db_logger.log_scalar('wfe_modes', rms_res_all_filt)
            db_logger.log_scalar('wfe_total', rms_res_all)

    def _save_loop_stats(self):
        '''
            Saving loop statistics to file

            Units are in nm
            For name of saved metrics, see evaluateSensorEstimate
            
            Saving (example)
                1. wfe_all_f_avg   : 'mode-limited' rms WFE of residual differential aberrations (all except SCAO residuals).
                                  `avg' because it use the WV integrated phase instead of the last instantaneous phase screen
                                  `f' : mode-limited
                2. wfe_qs_f        : 'mode-limieted' QS residuals (if water vapor enable, this is misleading)
                3. wfe_all         : same as 1., not mode-limited
                4. wfe_qs          : same as 2., not mode-limited
                5. input_wv_avg    : input WV phase disturbance, integrated of the sensor integration time (1/frame_rate)
        '''
        data = np.array(self._loop_stats)
        ll = ['%i']
        nb_entries = (len(self.loop_legend)-1)
        ll.extend( nb_entries * ['%f'])
        my_header = 'units are nm \n it'
        for i in range(nb_entries):
            my_header += ' \t '+ self.loop_legend[i+1]
        np.savetxt(os.path.join(self._directory, 'loopStats.csv'),
                   data,
                   header = my_header,
                   fmt=ll,
                   #header ='units are nm \n it \t wfe_all_f_avg \t wfe_qs_f \t wfe_all \t wfe_qs \t input_wv_avg ',
                   #fmt=['%i' , '%f', '%f', '%f', '%f', '%f'],
                   delimiter= '\t')

    def show(self, save_video=False):
        if self._axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
            self._axes = axes
        ax1, ax2, ax3, ax4 = self._axes
        # ax1 = plt.subplot(141, label='science')
        im1, _= imshow_norm(self.science_image, stretch=LogStretch(), ax=ax1)
        vmin = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 1)
        vmax = np.percentile(self.inst.phase_wv + self.inst.phase_ncpa, 99)
        inter = ManualInterval(vmin, vmax)
        #--
        # ax2 = plt.subplot(142, label='dist')
        phase = np.mean(self.inst.phase_wv_buffer + self.inst.phase_ncpa, axis=0)
        # masked = np.ma.masked_where(self.inst.aperture.shaped < 0.1, phase.shaped)
        # phase[self.inst.aperture < 0.1] = np.nan
        im2, _=imshow_norm(phase.shaped,
                           interval=inter, ax=ax2)
        # masked = np.ma.masked_where(self.inst.aperture.shaped < 0.1, self.inst.aperture.shaped)
        # ax2.imshow(masked)
        #--
        # ax3 = plt.subplot(143, label='wfs')
        # _dim = self.inst.pupilGrid.shape[0]
        im3, _=imshow_norm(-self.inst.phase_ncpa_correction.shaped * \
                    self.inst.aperture.shaped,
                    interval=inter, ax=ax3)
        # im3, _=imshow_norm(-self.inst.phase_ncpa_correction.reshape((_dim, _dim)) * \
        #                    self.inst.aperture.shaped,
        #                    interval=inter, ax=ax3)
        # im3, _=imshow_norm(self._wavefront.shaped * \
        #                    self.inst.aperture.shaped,
        #                    interval=inter, ax=ax3)
        #--
        # ax4 = plt.subplot(144, label='res')
        # phase = np.mean(self.inst.phase_wv_buffer + self.inst.phase_ncpa +
        #                    self.inst._next_phase_ncpa_correction, axis=0)
        phase = np.mean(self.inst.phase_wv_buffer + self.inst.phase_ncpa +
                           self.inst.phase_ncpa_correction, axis=0)
        im4, _=imshow_norm(self.inst.aperture.shaped *
                           phase.shaped,
                           interval=inter, ax=ax4)

        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax4.set_axis_off()
        ax2.set_title('static NCPA + WV')
        ax3.set_title('NCPA correction')
        ax4.set_title('Residuals')
        plt.tight_layout()
        if save_video:
            time.sleep(0.1)
            self._ims.append([im1, im2, im3, im4])    
        else:
            plt.draw()
            plt.pause(0.01)
        # time.sleep(0.1)
        # self._ims.append([im1, im2, im3, im4])    

    def save_video(self, fname='toto.mp4', fps=0.5):
        fig=plt.gcf()
        ani = animation.ArtistAnimation(fig, self._ims, interval=500,
                                        blit=False, repeat=False)
        FFwriter = animation.FFMpegWriter(fps=fps)
        ani.save(fname, writer=FFwriter)


    # def _store_phase_screens_to_file(self, i):
    #     '''
    #         Storing phase screens to fits file.
    #         Units of phase is nm

    #         Parameters
    #         ----------
    #          i : int
    #              index appended to the fits filename

    #         TODO populate the header with useful information
    #     '''
    #     conv2nm = self.inst.wavelength / (2*np.pi) * 1e9

    #     ncpa_correction = self.inst.phase_ncpa_correction * conv2nm
    #     ncpa_injected = (self.inst.phase_ncpa + self.inst.phase_wv) * conv2nm

    #     if type(ncpa_correction) == hcipy.Field:
    #         ncpa_correction = np.copy(ncpa_correction.shaped)
    #     if type(ncpa_injected) == hcipy.Field:
    #         ncpa_injected = np.copy(ncpa_injected.shaped)

    #     ncpa_residual = ncpa_injected + ncpa_correction

    #     filename = 'residual_ncpa_%s.fits'%i
    #     full_name = self._directory_phase + filename
    #     fits.writeto(full_name, ncpa_residual)
    #     hdr = fits.getheader(full_name)
    #     hdr.set('EXTNAME', 'NCPA_IN')
    #     fits.append(full_name, ncpa_injected, hdr)
    #     hdr = fits.getheader(full_name)
    #     hdr.set('EXTNAME', 'NCPA_COR')
    #     fits.append(full_name, ncpa_correction, hdr)