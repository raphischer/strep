import argparse
import os

from strep.load_experiment_logs import assemble_database


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir-root", default=None, help="")
    parser.add_argument("--logdir-merged", default='results_imagenet', help="")
    parser.add_argument("--output-tar-dir", default=None, help="")
    parser.add_argument("--property-extraxtors-module", default='properties_imagenet', help="")
    parser.add_argument("--database-dir", default='databases/imagenet_classification', help="")

    args = parser.parse_args()

    database = assemble_database(args.logdir_root, args.logdir_merged, args.output_tar_dir, args.property_extraxtors_module)
    if not os.path.isdir(args.database_dir):
        os.makedirs(args.database_dir)
    database.to_pickle(os.path.join(args.database_dir, 'database.pkl'))
