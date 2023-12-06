import argparse
import os

from mlprops.index_and_rate import rate_database, find_relevant_metrics, load_database
from mlprops.util import load_meta

DATABASES = {
    'ImageNetEff': 'databases/imagenet_classification/database.pkl',
    'RobustBench': 'databases/robustbench/database.pkl',
    'Forecasting': 'databases/dnn_forecasting/database.pkl',
    'Papers With Code': 'databases/paperswithcode/database.pkl',
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='interactive', choices=['interactive', 'paper_results'])
    # interactive exploration params
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    databases = {}
    for name, fname in DATABASES.items():
        print('LOADING', name)
        if not os.path.isfile(fname):
            raise RuntimeError('Could not find', fname)
        # load database
        database = load_database(fname)
        # load meta infotmation
        meta = load_meta(os.path.dirname(fname))
        # rate database
        database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
        rated_database, boundaries, real_boundaries, references = rate_database(database, meta)

        print(f'    database {name} has {rated_database.shape} entries')
        databases[name] = ( rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references )

    if args.mode == 'interactive':
        from mlprops.elex.app import Visualization
        app = Visualization(databases)
        app.run_server(debug=args.debug, host=args.host, port=args.port)

    if args.mode == 'paper_results':
        from paper_results import create_all
        create_all(databases)