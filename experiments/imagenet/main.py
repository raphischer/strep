import argparse
import os

import mlflow
import pandas as pd
from codecarbon import OfflineEmissionsTracker

def print_colored_block(message, ok=True):
    col = '\033[92m' if ok else '\033[91m'
    print(col + (u"\u2588"*60 + '\n')*4 + '\n')
    print(f"{message}\n")
    print((u"\u2588"*60 + '\n')*4 + '\033[0m')

model_subset_sizes = {
    'ConvNeXtBase': 100,
    'ResNet152V2': 2,
    'InceptionResNetV2': 4,
    'EfficientNetB7': 10,
    'EfficientNetB6': 8,
    'EfficientNetB5': 6,
    'EfficientNetB4': 3,
    'DenseNet201': 2,
    'ConvNeXtSmall': 2
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference benchmarking with keras models on ImageNet")
    # data and model input
    parser.add_argument("--experiment", default="/home/fischer/repos/mlprops/experiments/imagenet/")
    parser.add_argument("--model", default="EfficientNetB6")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--datadir", default="/data/d1/fischer_diss/imagenet")
    parser.add_argument("--measure_power_secs", default=1)
    parser.add_argument("--nogpu", default=0)
    parser.add_argument("--subset", default=0)
    args = parser.parse_args()

    mlflow.log_dict(args.__dict__, 'config.json')
    
    # init
    if args.nogpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from util import load_data_and_model
    model, ds, meta = load_data_and_model(args.datadir, args.model, nogpu=args.nogpu)
    meta['dataset'] = 'ImageNet (ILSVRC 2012)'
    meta['task'] = 'Inference'
    n_samples = 50000

    # take a snippet of the data, if only testing a subset (e.g., for energy profiling)
    if args.subset:
        if args.nogpu: # on CPU, models are about 10x slower
            ds = ds.take(len(ds) // 10)
            n_samples = n_samples // 10
        if args.model in model_subset_sizes: # take less data for big models
            ds = ds.take(len(ds) // model_subset_sizes[args.model])
            n_samples = n_samples // model_subset_sizes[args.model]

    # evaluate on validation
    tracker = OfflineEmissionsTracker(measure_power_secs=args.measure_power_secs, log_level='warning', country_iso_code="DEU")
    tracker.start()
    print_colored_block(f'STARTING ENERGY PROFILING FOR   {args.model.upper()}   on   {"CPU" if args.nogpu else "GPU"}')
    eval_res = model.evaluate(ds, return_dict=True)
    print_colored_block(f'STOPPING ENERGY PROFILING FOR   {args.model.upper()}   on   {"CPU" if args.nogpu else "GPU"}', ok=False)
    tracker.stop()

    if not args.subset:
        # evaluate robustness
        _, corr, _ = load_data_and_model('/data/d1/fischer_diss/imagenet', args.model, variant='corrupted_sample', batch_size=meta["batch_size"])
        corr_res = model.evaluate(corr, return_dict=True)
        for key, val in corr_res.items():
            eval_res[f'corr_{key}'] = val

        # assess model file size
        modelfile = 'model.weights.h5'
        model.save_weights(modelfile)
        eval_res['fsize'] = os.path.getsize(modelfile)
        os.remove(modelfile)

    # assess resource consumption
    emissions = 'emissions.csv'
    emission_data = pd.read_csv('emissions.csv').to_dict()
    eval_res['running_time'] = emission_data['duration'][0] / n_samples
    eval_res['power_draw'] = emission_data['energy_consumed'][0] * 3.6e6 / n_samples
    eval_res['parameters'] = model.count_params()

    # log results
    for key, val in eval_res.items():
        mlflow.log_metric(key, val)
    for key, val in meta.items():
        mlflow.log_param(key, val)
    mlflow.log_artifact(emissions)
    
    # cleanup
    os.remove(emissions)
    print(eval_res)
