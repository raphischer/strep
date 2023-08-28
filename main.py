import argparse
import os
import pandas as pd

from mlprops.index_and_rate import rate_database, find_relevant_metrics
from mlprops.util import load_meta

DATABASES = {
    'DNN Forecasting': 'databases/sklearn_openml_classification/database.pkl',
    # 'ImageNet Classification': 'databases/imagenet_classification/database.pkl',
    'RobustBench': 'databases/robustbench/database.pkl',
    # 'Sklearn Classification': 'databases/sklearn_openml_classification/database.pkl'
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='interactive', choices=['interactive'])
    # interactive exploration params
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    databases = {}
    for name, fname in DATABASES.items():
        if not os.path.isfile(fname):
            raise RuntimeError('Could not find', fname)
        # load database
        database = pd.read_pickle(fname)
        # load meta infotmation
        meta = load_meta(os.path.dirname(fname))
        # rate database
        rated_database, boundaries, real_boundaries, references = rate_database(database, meta)
        metrics, xaxis_default, yaxis_default = find_relevant_metrics(database)
        
        print(f'Database {name} has {rated_database.shape} entries')
        databases[name] = ( rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references )

    if args.mode == 'interactive':
        from mlprops.elex.app import Visualization
        app = Visualization(databases)
        app.run_server(debug=args.debug, host=args.host, port=args.port)
