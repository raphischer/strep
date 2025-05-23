import json
import os
import random as python_random
import re
import pathlib

import numpy as np
import pandas as pd


def find_sub_db(database, dataset=None, task=None, environment=None, model=None):
    if dataset is not None:
        database = database[database['dataset'] == dataset]
    if task is not None:
        database = database[database['task'] == task]
    if environment is not None:
        database = database[database['environment'] == environment]
    if model is not None:
        database = database[database['model'] == model]
    return drop_na_properties(database) # drop columns with full NA


def load_meta(directory=None):
    if directory is None:
        directory = os.getcwd()
    if os.path.isfile(directory):
        directory = os.path.dirname(directory)
    meta = {'properties': {}}
    for fname in os.listdir(directory):
        re_match = re.match('meta_(.*).json', fname)
        if re_match:
            meta[re_match.group(1)] = read_json(os.path.join(directory, fname))
    if len(meta['properties']) == 0:
        print('Could not find any meta information - assuming all numeric properties relate to Quality and want to be maximized.')
    meta['meta_dir'] = os.path.abspath(directory)
    return meta


def loopup_task_ds_metrics(val_bounds):
    return { (task, ds): list(prop_bounds.keys()) for (task, ds, _), prop_bounds in val_bounds.items() }


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
    

def fill_meta(summary, meta):
    for property, value in list(summary.items()):
        try:
            summary[property] = meta[property][value]
        except KeyError:
            pass
    return summary


def basename(directory):
    if len(os.path.basename(directory)) == 0:
        directory = os.path.dirname(directory)
    return os.path.basename(directory)


def write_json(filepath, dict):
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w') as outfile:
        json.dump(dict, outfile, indent=4, cls=PatchedJSONEncoder)


def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)


def read_txt(filepath):
    with open(filepath, 'r') as reqf:
        return [line.strip() for line in reqf.readlines()]
    

def read_csv(filepath):
    # use dumps and loads to make sure the log can be used with json (all keys in dict should be strings!)
    return json.loads(json.dumps(pd.read_csv(filepath).to_dict()))


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
        if isinstance(obj, pathlib.PosixPath):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def fix_seed(seed):
    if seed == -1:
        seed = python_random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    python_random.seed(seed)
    return seed


def prop_dict_to_val(df, key='value'):
    df = df.dropna(how='all', axis=1)
    properties = [col for col in df.columns if f'{col}_index' in df.columns]
    if key == 'value':
        return df[properties]
    elif key == 'index':
        return df[[f'{col}_index' for col in properties]]
    else:
        raise RuntimeError('unknown key', key)


def drop_na_properties(df, reduce_to_index=False):
    if reduce_to_index:
        df = prop_dict_to_val(df)
    valid_cols = df.dropna(how='all', axis=1).columns
    return df[valid_cols]


def weighted_median(values, weights):
    assert np.isclose(weights.sum(), 1), "Weights for weighted median should sum up to one"
    asort = values.argsort()
    values, cumw = values[asort], np.cumsum(weights[asort])
    for i, (cw, v) in enumerate(zip(cumw, values)):
        if cw == 0.5:
            return np.average([v, values[i + 1]])
        if cw > 0.5 or (cw < 0.5 and cumw[i + 1] > 0.5):
            return v
    raise RuntimeError
