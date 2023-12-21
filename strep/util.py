from datetime import datetime
import itertools
import json
import os
import random as python_random
import sys
import pkg_resources
import re

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

from strep.monitoring import log_system_info


def identify_all_correlations(db, all_metrics, scale='index'):
    corr = {}
    for ds_task, data in db.groupby(['dataset', 'task']):
        # init correlation table
        metrics = all_metrics[ds_task]
        corr[ds_task] = (np.full((metrics.size, metrics.size), fill_value=np.nan), metrics)
        np.fill_diagonal(corr[ds_task][0], 1)
        # assess correlation between properties
        props = prop_dict_to_val(data[metrics], scale)
        for idx_a, idx_b in itertools.combinations(np.arange(metrics.size), 2):
            if scale == 'index': # these originally were nan values!
                props[props == 0] = np.nan
            cols = props.iloc[:, [idx_a, idx_b]].dropna().values
            if cols.size > 4:
                corr[ds_task][0][idx_a, idx_b] = pearsonr(cols[:,0], cols[:,1])[0]
                corr[ds_task][0][idx_b, idx_a] = corr[ds_task][0][idx_a, idx_b]
    return corr


def identify_correlation(db):
    correlation = np.zeros((len(db.columns), len(db.columns)))
    for col_a, col_b in itertools.combinations(np.arange(len(db.columns)), 2):
        correlation[col_a, col_b] = pearsonr(db.iloc[:, col_a], db.iloc[:, col_b])[0]
        correlation[col_b, col_a] = correlation[col_a, col_b]
    return correlation, db.columns.tolist()


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
    if key == 'name' and '_index' in element_name:
        return f'{element_name.replace("_index", "").capitalize()} Index'
    try:
        if subdict is not None and subdict in meta:
            found = meta[subdict][element_name]
        else:
            found = meta[element_name]
        if len(key) > 0:
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
        if pd.isnull(obj):
            return None
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


def prop_dict_to_val(df, key='value'):
    return df.map(lambda val: val[key] if isinstance(val, dict) and key in val else val)


def drop_na_properties(df):
    valid_cols = prop_dict_to_val(df).dropna(how='all', axis=1).columns
    return df[valid_cols]


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
