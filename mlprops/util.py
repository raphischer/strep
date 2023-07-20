from datetime import datetime
import json
import os
import random as python_random
import sys
import pkg_resources
import re

import numpy as np
import pandas as pd

from mlprops.monitoring import log_system_info


def load_meta(directory=None):
    if directory is None:
        directory = os.getcwd()
    meta = {}
    for fname in os.listdir(directory):
        re_match = re.match('meta_(.*).json', fname)
        if re_match:
            meta[re_match.group(1)] = read_json(os.path.join(directory, fname))
    return meta


def lookup_meta(meta, element_name, key='name', subdict=None):
    try:
        if subdict is not None and subdict in meta:
            found = meta[subdict][element_name]
        else:
            found = meta[element_name]
        if key is not None:
            return found[key]
        return found
    except KeyError:
        return element_name


def basename(directory):
    if len(os.path.basename(directory)) == 0:
        directory = os.path.dirname(directory)
    return os.path.basename(directory)


def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)


def read_txt(filepath):
    with open(filepath, 'r') as reqf:
        return [line.strip() for line in reqf.readlines()]


class PatchedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)


def fix_seed(seed):
    if seed == -1:
        seed = python_random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    python_random.seed(seed)
    return seed


def create_output_dir(dir=None, prefix='', config=None):
    if dir is None:
        dir = os.path.join(os.getcwd())
    # create log dir
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if len(prefix) > 0:
        timestamp = f'{prefix}_{timestamp}'
    dir = os.path.join(dir, timestamp)
    while os.path.exists(dir):
        dir += '_'
    os.makedirs(dir)
    # write config
    if config is not None: 
        with open(os.path.join(dir, 'config.json'), 'w') as cfg:
            config['timestamp'] = timestamp.replace(f'{prefix}_', '')
            json.dump(config, cfg, indent=4)
    # write installed packages
    with open(os.path.join(dir, 'requirements.txt'), 'w') as req:
        for v in sys.version.split('\n'):
            req.write(f'# {v}\n')
        for pkg in pkg_resources.working_set:
            req.write(f'{pkg.key}=={pkg.version}\n')
    log_system_info(os.path.join(dir, 'execution_platform.json'))
    return dir


class Logger(object):
    def __init__(self, fname='logfile.txt'):
        self.terminal = sys.stdout
        self.log = open(fname, 'a')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal