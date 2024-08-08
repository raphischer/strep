import argparse
import os

import mlflow

from codecarbon import OfflineEmissionsTracker

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference benchmarking with keras models on ImageNet")
    # data and model input
    parser.add_argument("--experiment", default="/home/fischer/repos/mlprops/experiments/imagenet/")
    parser.add_argument("--model", default="ResNet50")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--datadir", default="/data/d1/fischer_diss/imagenet")
    parser.add_argument("--measure_power_secs", default=1)
    parser.add_argument("--nogpu", default=0)
    args = parser.parse_args()
    mlflow.log_dict(args.__dict__, 'config.json')
    
    # init
    if args.nogpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from util import load_data_and_model, CPU_SIZES, GPU_SIZES
    batch_size = CPU_SIZES[args.model] if args.nogpu else GPU_SIZES[args.model]
    model, ds, info = load_data_and_model('/data/d1/fischer_diss/imagenet', args.model, batch_size)

    # evaluate
    emissions = 'emissions.csv'
    tracker = OfflineEmissionsTracker(measure_power_secs=args.measure_power_secs, log_level='warning', country_iso_code="DEU")
    tracker.start()
    eval_res = model.evaluate(ds, return_dict=True)
    tracker.stop()

    # log results
    for key, val in eval_res.items():
        mlflow.log_metric(key.replace('sparse_', '').replace('categorical_', '').replace('top_k', 'top_5'), val)
    mlflow.log_metric('parameters', model.count_params())
    mlflow.log_artifact(emissions)
    os.remove(emissions)
