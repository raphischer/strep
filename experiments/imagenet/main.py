import argparse
import os

import mlflow
import pandas as pd
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
    from util import load_data_and_model
    model, ds, meta = load_data_and_model(args.datadir, args.model)
    meta['dataset'] = 'ImageNet (ILSVRC 2012)'
    meta['task'] = 'Inference'

    # evaluate on validation
    tracker = OfflineEmissionsTracker(measure_power_secs=args.measure_power_secs, log_level='warning', country_iso_code="DEU")
    tracker.start()
    eval_res = model.evaluate(ds, return_dict=True)
    tracker.stop()

    # evaluate robustness
    _, corr, _ = load_data_and_model('/data/d1/fischer_diss/imagenet', args.model, variant='corrupted_sample', batch_size=meta["batch_size"])
    corr_res = model.evaluate(corr, return_dict=True)
    for key, val in corr_res.items():
        eval_res[f'corr_{key}'] = val

    # assess some additional information
    emissions, modelfile = 'emissions.csv', 'model.weights.h5'
    model.save_weights(modelfile)
    eval_res['fsize'] = os.path.getsize(modelfile)
    eval_res['parameters'] = model.count_params()
    emission_data = pd.read_csv('emissions.csv').to_dict()
    eval_res['running_time'] = emission_data['duration'][0] / 50000
    eval_res['power_draw'] = emission_data['energy_consumed'][0] * 3.6e6 / 50000

    # log results
    for key, val in eval_res.items():
        mlflow.log_metric(key, val)
    for key, val in meta.items():
        mlflow.log_param(key, val)
    mlflow.log_artifact(emissions)
    
    # cleanup
    os.remove(emissions)
    os.remove(modelfile)
