import argparse
import os
import time

from strep.index_and_rate import rate_database, find_relevant_metrics, load_database, prop_dict_to_val
from strep.util import load_meta
from strep.elex.app import Visualization

DATABASES = {
    # 'Papers With Code': 'databases/paperswithcode/database.pkl',
    # 'MetaQuRe': 'databases/metaqure/database.pkl',
    'ImageNetEff': 'databases/imagenet_classification/database.pkl',
    # 'RobustBench': 'databases/robustbench/database.pkl',
    # 'Forecasting': 'databases/xpcr/database.pkl'
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
    from strep.index_scale import scale_and_rate
    import numpy as np
    t0 = time.time()
    proc_new, boundaries = scale_and_rate(database, meta)
    t1 = time.time()
    rated_database, boundaries_old, real_boundaries, references = rate_database(database, meta)
    t2 = time.time()
    scaled_old = prop_dict_to_val(rated_database, 'index')
    rated_old = prop_dict_to_val(rated_database, 'rating')
    for prop in meta['properties'].keys():
        new = proc_new[f'{prop}_index'].dropna()
        if not np.all(np.isclose(scaled_old.loc[new.index,prop].values, new)):
            print(f'not all equal for {fname} {prop}')
    for comp in [col for col in rated_database.columns if '_index' in col]:
        if not np.all(np.isclose(rated_database[comp], proc_new[comp])):
            print(f'not all compound index vals equal for {fname} {comp}')
    print(f'    database {name} has {rated_database.shape} entries - new scaling took {t1-t0:5.3f}, old scaling took {t2-t1:5.3f}')
    database2, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
    return rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='interactive', choices=['interactive', 'paper_results'])
    parser.add_argument("--database", default=None)
    # interactive exploration params
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    if args.database is not None:
        DATABASES = {'CUSTOM': args.database}

    databases = {}
    for name, fname in DATABASES.items():
        print('LOADING', fname)
        databases[name] = preprocess_database(fname)
        # override defaults for robustbench
        if 'robustbench' in fname:
            for ds_task in databases[name][2].keys():
                databases[name][3][ds_task] = 'clean_acc' # x axis
                databases[name][4][ds_task] = 'autoattack_acc' # y axis

    app = Visualization(databases, use_pages=True, pages_folder='')
    server = app.server
        
    app.run_server(debug=args.debug, host=args.host, port=args.port)
