import argparse

from strep.index_scale import load_database, scale_and_rate
from strep.elex.app import Visualization

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", default='databases/imagenet_classification/database.pkl')
    args = parser.parse_args()

    # load database and meta information (if available)
    database, meta = load_database(args.fname)
    # index-scale and rate database
    rated_database = scale_and_rate(database, meta)
    # start the interactive exploration tool
    app = Visualization(rated_database)
    app.run_server()
