'''
Source: VLADI software
https://github.com/GillesOrban/VLADI
'''
# reads data from file, can be called from data object or from filename

import h5py
import os
import yaml

class readTools:

    def read_h5(self, filename=None, dataObj=None):
        if dataObj is not None:
            if dataObj.mode is None:
                raise ValueError('grabFrames() is not run yet on the data object')
            filename = dataObj.config["dataset_file"] + '{}/ds_{}.h5'.format(dataObj.mode, dataObj.testLabel)
        elif filename is None:
            raise ValueError('Either filename or dataObj must be specified')

        db = {}
        with h5py.File(filename, 'r') as f:
            # cube = f['frame'][:]
            # zernikeCoeff = f['zernikeCoeff'][:]
            for key in f.keys():
                db[key] = f[key][:]
            attrs = dict(f.attrs.items())

        # converting strings to actual values
        if 'tip_tilt' in attrs.keys(): 
            attrs['tip_tilt'] = attrs['tip_tilt'] == 'True'
        
        return db, attrs

    def read_conf(self, conf_file):
        assert os.path.isfile(conf_file), "The path to the configuration file specified does not exist: {}".format(conf_file)

        with open(conf_file, 'r') as cf:
            try:
                conf = yaml.safe_load(cf)
            except yaml.YAMLError as err:
                print(err)
        return conf