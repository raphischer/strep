import os
import json
import tarfile
import importlib

import pandas as pd
import numpy as np

from mlprops.monitoring import aggregate_monitoring_log
from mlprops.util import basename, PatchedJSONEncoder, read_json, read_txt


#############################
####     log loading     ####
#############################


def read_monitoring(filepath):
    return aggregate_monitoring_log(filepath)


def read_csv(filepath):
    # use dumps and loads to make sure the log can be used with json (all keys in dict should be strings!)
    return json.loads(json.dumps(pd.read_csv(filepath).to_dict()))


def read_log_directory(directory):
    reader_methods = {name: func for name, func in globals().items() if name.startswith('read_')}
    res = {'directory_name': basename(directory)}
    # read all files
    for filename in os.listdir(directory):
        fbase, ext = os.path.splitext(filename)
        reader_method = f'read_{ext[1:]}'
        if reader_method in reader_methods:
            res[fbase] = reader_methods[reader_method](os.path.join(directory, filename))
    return res


def process_directory(directory, outout_logdir_merged=None, output_tar_dir=None):
    print('Processing', directory)
    # create summary
    if outout_logdir_merged is not None: 
        if not os.path.isdir(outout_logdir_merged):
            os.makedirs(outout_logdir_merged)
        agglog_name = os.path.join(outout_logdir_merged, basename(directory) + '.json')
        # load if already exists
        if os.path.isfile(agglog_name):
            res = read_json(agglog_name)
        else:
            res = read_log_directory(directory)
            if outout_logdir_merged is not None:
                res['full_log'] = os.path.join(outout_logdir_merged, basename(directory) + '.tar.gz')
            with open(agglog_name, 'w') as agglog:
                json.dump(res, agglog, indent=4, cls=PatchedJSONEncoder)
    else:
        res = read_log_directory(directory)
    # create tar
    if output_tar_dir is not None:
        if not os.path.isdir(output_tar_dir):
            os.makedirs(output_tar_dir)
        log_tar_name = os.path.join(output_tar_dir, basename(directory) + '.tar.gz')
        if not os.path.exists(log_tar_name):
            with tarfile.open(log_tar_name, 'w:gz') as tar:
                for fname in os.listdir(directory):
                    tar.add(os.path.join(directory, fname))
    return res


#############################
####   log aggregation   ####
#############################

def aggregate_log(log, property_extractors):
    log_name = log['directory_name']
    agg_log = {'log_name': log_name}
    for task, extractors in property_extractors.items():
        if log_name.startswith(task) or task == 'meta':
            # use extractor functions
            for key, func in extractors.items():
                try:
                    agg_log[key] = func(log)
                except KeyError as e:
                    try:
                        conf = f'{log["config"]["model"]} on {log["config"]["dataset"]}'
                    except KeyError:
                        conf = 'n.a.'
                    print(f'Error in assessing {key:<15} from {log_name} - {conf}!')
                    agg_log[key] = np.nan
    return agg_log


def merge_database(database):
    database['configuration'] = database.aggregate(lambda row: ' - '.join([row['task'], row['dataset'], row['model']]), axis=1)
    database['environment'] = database.aggregate(lambda row: ' - '.join([str(row['architecture']), str(row['software'])]), axis=1)
    grouped = database.groupby(['configuration', 'environment'])
    grouped_results = grouped.first() # take first occurence as a start
    mean_values = grouped.mean()
    grouped_results.update(mean_values) # TODO also merge the individual log directories into list
    grouped_results['n_results'] = grouped.size()
    return grouped_results.reset_index()


def aggregate_logs(logs, property_extractors_module):
    # import extractors
    if property_extractors_module is None:
        property_extractors_module = 'properties'
    try:
        property_extractors = importlib.import_module(property_extractors_module).PROPERTIES
    except (AttributeError, ImportError) as e:
        raise RuntimeError(f'Error when trying to import aggregators from {property_extractors_module} module!')

    if not isinstance(logs, list):
        # directory given instead of logs
        if not os.path.isdir(logs):
            raise RuntimeError('Please pass a list of merged logs, or the directory where respective .json logs can be found!')
        # parse all logs in directory
        lognames = [os.path.join(logs, fname) for fname in os.listdir(logs) if fname.endswith('.json')]
        logs = [read_json(fname) for fname in lognames]
    
    # aggregate properties per log, build database, and merge to single entries per (configuration X environment) combination
    agg_logs = [ aggregate_log(log, property_extractors) for log in logs ]
    aggregated_logs = pd.DataFrame.from_records(agg_logs)
    merged_database = merge_database(aggregated_logs)

    return merged_database


def assemble_database(logdir_root=None, logdir_merged=None, output_tar_dir=None, property_extractors_module=None):
    # process
    logs = []
    if logdir_root is not None:
        for dir in sorted(os.listdir(logdir_root)):
            log = process_directory(os.path.join(logdir_root, dir), logdir_merged, output_tar_dir)
            logs.append(log)

    if logdir_merged is not None:
        if not os.path.isdir(logdir_merged):
            raise RuntimeError(f'Invalid logdir_merged option "{logdir_merged}" (not a directory)!')
        logs = [] # re-init and read from merged dir, to avoid duplicates
        for fname in os.listdir(logdir_merged):
            try:
                logs.append( read_json(os.path.join(logdir_merged, fname)) )
            except Exception as e:
                print(f'Could not load log {fname}! {e}')

    # aggregate
    database = aggregate_logs(logs, property_extractors_module)

    return database


def find_sub_database(database, dataset=None, task=None, environment=None):
    if dataset is not None:
        database = database[database['dataset'] == dataset]
    if task is not None:
        database = database[database['task'] == task]
    if environment is not None:
        database = database[database['environment'] == environment]
    return database
