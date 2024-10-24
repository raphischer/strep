import argparse
import os
import sys
import time

import numpy as np
import mlflow
import pandas as pd
from codecarbon import OfflineEmissionsTracker

from util import print_colored_block
from batch_sizes import lookup_batch_size, find_ideal_batch_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference benchmarking with keras models on ImageNet")
    # data and model input
    parser.add_argument("--experiment", default="/home/fischer/repos/mlprops/experiments/imagenet/")
    parser.add_argument("--model", default="DenseNet121")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--datadir", default="/data/d1/fischer_diss/imagenet")
    parser.add_argument("--measure_power_secs", default=0.5)
    parser.add_argument("--nogpu", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=0, help="number of seconds to profile model on a subset of the data -- 0 process complete")
    args = parser.parse_args()
    mlflow.log_dict(args.__dict__, 'config.json')
    
    # if required, disable gpu
    if args.nogpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # identify batch_size and load data
    batch_size = 4 # lookup_batch_size(args.model) or find_ideal_batch_size(args.model, args.nogpu, args.datadir)
    from data_and_model_loading import load_data_and_model # import inits tensorflow, so only import now
    model, ds, meta = load_data_and_model(args.datadir, args.model, batch_size=batch_size)
    meta['dataset'] = 'ImageNet (ILSVRC 2012)'
    meta['task'] = 'Inference'
    for key, val in meta.items():
        mlflow.log_param(key, val)
    model.evaluate(ds.take(1)) # init inference (often has some temporal overhead)
    n_samples = 50000

    # given limit for evaluation, so only take a subset of the data
    if args.seconds:
        t0 = time.time()
        model.evaluate(ds.take(1)) # first test a single batch
        t1 = time.time()
        test_n = max(5, np.round(args.seconds / (4 * (t1-t0)))) # test min of five batches, but if very fast, 1/4 of the time limit
        model.evaluate(ds.take(test_n + 1))
        t_single_without_init = (time.time() - t1 - (t1 - t0)) / test_n # remove overhead and calc per sample time
        n_batches = np.round(args.seconds / t_single_without_init)
        n_samples = n_batches * meta['batch_size']
        while n_batches > len(ds): # for very fast models, we maybe need to repeat the dataset several times
            ds = ds.concatenate(ds)
        ds = ds.take(n_batches)
        print(f'Processing {n_batches} batches, per batch expected runtime {t_single_without_init:.4f}s, len ds {len(ds)}')

    # evaluate on validation
    mlflow.log_param('n_samples', n_samples)
    tracker = OfflineEmissionsTracker(measure_power_secs=args.measure_power_secs, log_level='warning', country_iso_code="DEU")
    tracker.start()
    print_colored_block(f'STARTING ENERGY PROFILING FOR   {args.model.upper()}   on   {"CPU" if args.nogpu else "GPU"}')
    eval_res = model.evaluate(ds, return_dict=True)
    print_colored_block(f'STOPPING ENERGY PROFILING FOR   {args.model.upper()}   on   {"CPU" if args.nogpu else "GPU"}', ok=False)
    tracker.stop()

    if not args.seconds:
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
    eval_res['time_total'] = emission_data['duration'][0]
    eval_res['running_time'] = emission_data['duration'][0] / n_samples
    eval_res['power_draw'] = emission_data['energy_consumed'][0] * 3.6e6 / n_samples
    eval_res['parameters'] = model.count_params()

    # log results
    for key, val in eval_res.items():
        mlflow.log_metric(key, val)
    mlflow.log_artifact(emissions)
    
    # cleanup
    os.remove(emissions)
    print(eval_res)
    sys.exit(0)
