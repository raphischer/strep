import argparse
import os
import pandas as pd

from mlprops.index_and_rate import rate_database
from mlprops.util import load_meta


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='interactive', choices=['interactive'])
    parser.add_argument("--database-dir", default="databases/imagenet_classification", help="directory with database.pkl and meta info inside")
    # interactive exploration params
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    # load database
    database_file = os.path.join(args.database_dir, 'database.pkl')
    if not os.path.isfile(database_file):
        raise RuntimeError('Could not find', database_file)
    database = pd.read_pickle(database_file)

    # load meta infotmation
    meta = load_meta(args.database_dir)

    # rate database
    rated_database, boundaries, real_boundaries, _ = rate_database(database, meta)
    print(f'Database constructed has {rated_database.shape} entries')

    if args.mode == 'interactive':
        from mlprops.elex.app import Visualization
        app = Visualization(rated_database, boundaries, real_boundaries, meta)
        app.run_server(debug=args.debug, host=args.host, port=args.port)
