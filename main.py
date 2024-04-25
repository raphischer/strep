import argparse
import os

from strep.index_and_rate import rate_database, find_relevant_metrics, load_database
from strep.util import load_meta

DATABASES = {
    'ImageNetEff': 'databases/imagenet_classification/database.pkl',
    'RobustBench': 'databases/robustbench/database.pkl',
    'Forecasting': 'databases/dnn_forecasting/database.pkl',
    'Papers With Code': 'databases/paperswithcode/database.pkl',
}

def preprocess_database(fname):
    if not os.path.isfile(fname):
        raise RuntimeError('Could not find', fname)
    # load database
    database = load_database(fname)
    if 'paperswithcode' in fname:
        database['dataset'] = database['dataset'].map(lambda val: 'KITTI' if val == 'kitti-depth-completion' else val)
    # load meta infotmation
    meta = load_meta(os.path.dirname(fname))
    # rate database
    database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
    rated_database, boundaries, real_boundaries, references = rate_database(database, meta)
    print(f'    database {name} has {rated_database.shape} entries')
    return rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references

databases = {}

for name, fname in DATABASES.items():
    print('LOADING', fname)
    databases[name] = preprocess_database(fname)
    # override defaults for robustbench
    if 'robustbench' in fname:
        for ds_task in databases[name][2].keys():
            databases[name][3][ds_task] = 'clean_acc' # x axis
            databases[name][4][ds_task] = 'autoattack_acc' # y axis

from strep.elex.app import Visualization
app = Visualization(databases)
server = app.server

if __name__ == '__main__':    
    app.run_server(debug=False)
