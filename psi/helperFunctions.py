from time import time
from functools import wraps
import os
import shutil
import datetime
import getpass
# import colored
import json
from argparse import ArgumentParser
from colorama import Fore


class LazyLogger(object):
    def __init__(self, name=''):
        self._name = name
        pass

    def debug(self, msg):
        print(("{0} - " + Fore.CYAN +
               "[Debug]: " + Fore.RESET).format(self._name) , msg)

    def notice(self, msg):
        print(("{0} - " + Fore.GREEN +
               "[Info]: " + Fore.RESET).format(self._name) , msg)

    def info(self, msg):
        self.notice(msg)

    def warning(self, msg):
        print(("{0} - " + Fore.YELLOW +
               "[Warning]: "+ Fore.RESET).format(self._name) , msg)

    def warn(self, msg):
        print(("{0} - " + Fore.YELLOW +
               "[Warning]: " + Fore.RESET).format(self._name) , msg)

    def error(self, msg):
        print(("{0} - " + Fore.RED +
               "[Error]: " + Fore.RESET).format(self._name) , msg)


def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it


def build_directory_name(config_file, basedir):
    rawfilename = os.path.splitext(os.path.basename(config_file))[0]
    bdir = basedir
    now = datetime.datetime.now()
    directory_name =  getpass.getuser()
    directory_name += '_' + rawfilename
    directory_name += '_' + now.strftime("%Y-%m-%dT%H:%M:%S")
    directory_name += '/'
    directory_name = os.path.join(bdir,directory_name)
    return directory_name

def copy_cfgFileToDir(directory, config_file):
    destination_file = directory + '/config/' + os.path.basename(config_file)
    os.mkdir(directory + '/config/')
    shutil.copyfile(config_file, destination_file)


def dump_config_to_text_file(file_name, cfg):
    '''
    PARAMETERS
        file_name : str
        cfg : namespace
    '''
    with open(file_name, 'w') as f:
        json.dump(sorted(cfg.__dict__), f, indent=2)


def read_config_from_text_file(file_name):
    '''
    Read the config from a text file written with 'dump_config_to_text_file'

    RETURNS
        a namespace
    '''
    parser = ArgumentParser()
    args = parser.parse_args()
    with open(file_name, 'r') as f:
        args.__dict__ = json.load(f)
    return args

