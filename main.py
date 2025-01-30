import argparse

from strep.index_scale import load_database, scale_and_rate
from strep.elex.app import Visualization

DATABASES = {
    'ImageNetEff22': 'databases/imagenet_classification/database.pkl',
    'EdgeAccUSB': 'databases/edge_acc/database.pkl',
    'XPCR': 'databases/xpcr/database.pkl',
    'XPCR_WITH_COMPS': 'databases/xpcr/database.pkl',
    'MetaQuRe': 'databases/metaqure/database.pkl',
    'PWC': 'databases/paperswithcode/database.pkl',
    # 'PWC_FULL': 'databases/paperswithcode/database_complete.pkl', # disabled because it takes very long to load
    'RobBench': 'databases/robustbench/database.pkl',
}

CUSTOM_DB_CHANGES = {
    "XPCR": lambda db: db[~db["model"].isin(["autokeras", "autogluon", "autosklearn"])]
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", default=None)
    args = parser.parse_args()
    databases = {}
    
    if args.custom: # load custom database and meta information (if available)
        database, meta = load_database(args.custom)
        # index-scale and rate database
        databases = scale_and_rate(database, meta)

    else: # load pre-defined databases
        for name, fname in DATABASES.items():
            print('LOADING', fname)
            database, meta = load_database(fname)
            if name in CUSTOM_DB_CHANGES:
                database = CUSTOM_DB_CHANGES[name](database)
            databases[name] = scale_and_rate(database, meta)

    # start the interactive exploration tool
    app = Visualization(databases)
    app.run_server()
