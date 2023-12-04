'''
This file follow the implementation in VLADI ``phase_prediction.py''

'''
import os
import numpy as np
import json

# ML import
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import psi.deep_wfs.src.Transforms as custom_transforms
import psi.deep_wfs.src.Resnet as resnet
import psi.deep_wfs.utils.dataset_format_pytorch as datapp
import psi.deep_wfs.utils.read_data as rt
from psi.deep_wfs.utils.dataset_format_pytorch import normalization
from psi.helperFunctions import LazyLogger

# rt = readTools()


class WrappedModel(nn.Module):
    '''
    Need to be called before loading the weights if several GPUs were used for training
    '''
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)

class dataInfer:
    '''
    This class provides the functionality to use a given ResNet model 
    and to perform a forward pass (prediction), via the method ``infer''
    '''
    def __init__(self, logger=LazyLogger('deep_infer')):
        # TODO option to assign 2 GPUs (or more)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=None
        self.logger = logger

    def setup(self, conf_file=None, model_data_path=None):
        _config, _data_info = self.setConfig(conf_file,  model_data_path=model_data_path)
        self.config = _config
        self.data_info = _data_info
        if not 'nb_modes' in self.data_info:
            self.logger.warning('Warning: data info incomplete, setting nb_modes from config')
            self.data_info['nb_modes'] = self.config['nb_modes']
        if not 'channels' in self.data_info:
            self.logger.warning('Warning: data info incomplete, setting channels from config')
            self.data_info['channels'] = 1
        if not 'wavelength' in self.data_info:
            self.logger.warning('Warning: data info incomplete, setting wavelength from config')
            self.data_info['wavelength'] = 1

    def infer(self, psfs, conf_file=None):
        '''
        PARAMETERS
        ----------
        psfs  : 3d numpy array
            if a single PSF, dimension should be (1, xdim, ydim).
        config_file :  string

        RETURNS
        -------
        zernike_vector  : 2d numpy array
            1st dim is correspond to number of entries (==1 if a single PSF)
            2nd dim is the number of modes
        '''
        # _config, _data_info = self.setConfig(conf_file)
        _data_info = self.data_info
        _config = self.config
        _data_info['nb_samples'] = psfs.shape[0]

        if self.model is None:
            self.logger.info('Loading CNN model for the 1st time')
            self.model = self.load_model(_config, _data_info)
        
        # Inference
        dataset_input = {"psfs_1": psfs}
        dataset_normalized = normalization(dataset_input, _data_info)

        zernike_predicted = torch.zeros((_data_info['nb_samples'],
                                         _data_info['nb_modes']))
        for i in range(_data_info['nb_samples']):
            zernike_predicted[i, :] = self.model(dataset_normalized[i]["image"].unsqueeze(0))

        # Conversion to Numpy arrays
        rad2nm = _data_info["wavelength"] / (2 * np.pi)
        zernike_vector = rad2nm * zernike_predicted.detach().numpy()

        return zernike_vector

    def load_model(self, config, data_info):
        '''
        copy/paste from VLADI

        TODO add option to simply load ImageNet ResNet model (w/o loading specific trained weights)
        '''


        # Load the model architecture (without trained weights):
        model = resnet.Net(n_channels_in=1,
                        n_channels_out=data_info["nb_modes"],
                        resnet_archi=config["model_type"]).eval()  # Model set in evaluation mode (discard batch normalization)

        # Read and load the trained weights from existing file
        # if not os.path.isfile(config["model_dir"] + "model.pth"):
        #     raise FileNotFoundError("Model not found in the specified directory: " + config["model_dir"])

        if os.path.isfile(config["model_dir"] + "model.pth"):
            model_loc = config["model_dir"] + "model.pth"

            state_dict = torch.load(model_loc, map_location=self.device)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                # Necessary for loading the weights if trained with several GPUs
                model = WrappedModel(model)
                model.load_state_dict(state_dict)
        else:
            self.logger.warning('No existing trained weights -- using the default ResNet')

        return model
    
    # def normalization(self, dataset, data_info):
    #     transfo_list = [custom_transforms.Normalize()]
    #     dataset_norm = datapp.psf_dataset(dataset=dataset,
    #                                     data_info=data_info,
    #                                     transform=transforms.Compose(transfo_list))
        
    #     return dataset_norm
    
    def setConfig(self, conf_file=None, model_data_path=None):
        if conf_file is None:
            conf_file = os.path.dirname(__file__) + "/config/inference_config.yml"
        else:
            #_config = conf_file
            pass
        _config = rt.read_conf(conf_file=conf_file)
        if model_data_path is not None:
            _config['model_dir'] = model_data_path

        # print(_config)
        # # if not os.path.isfile(_config["model_dir"] + "model.pth"):
        # if not os.path.isfile(_config["model_fname"] + "model.pth"):
        #     raise FileNotFoundError("Model not found in the specified directory: " + _config["model_dir"])
        
        if os.path.isfile(_config["model_dir"] + "/model.pth"):
            # read data_info.json
            with open(os.path.dirname(_config["model_dir"]) + \
                    "/data_info.json", "r") as f:
                _data_info = json.load(f)
        else:
            #_data_info = None
            self.logger.warning('No data info')
            _data_info = {}


        return _config, _data_info