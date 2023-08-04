import json


PROPERTIES = {
    'meta': {
        'task': lambda log: log['directory_name'].split('_')[0],
        'dataset': lambda log: log['config']['dataset'] + '_' + str(log['config']['ds_seed']) if log['config']['ds_seed'] != -1 else log['config']['dataset'],
        'model': lambda log: log['config']['model'],
        'architecture': lambda log: extract_architecture(log),
        'software': lambda log: extract_software(log),
    },

    'train': {
        'train_running_time': lambda log: log['emissions']['duration']['0'],
        'train_power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6
    },
    
    'infer': {
        'running_time': lambda log: log['emissions']['duration']['0'] / log['validation_results']['num_samples'],
        'power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6 / log['validation_results']['num_samples'],
        'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['RMSE'],
        'MAPE': lambda log: log['validation_results']['metrics']['aggregated']['MAPE'],
        'MASE': lambda log: log['validation_results']['metrics']['aggregated']['MASE'],
        'parameters': lambda log: log['validation_results']['model']['params'],
        'fsize': lambda log: log['validation_results']['model']['fsize'],
    }
}


def extract_architecture(log):
    with open('meta_environment.json', 'r') as meta:
        processor_shortforms = json.load(meta)['processor_shortforms']
    if 'GPU' in log['execution_platform']:
        n_gpus = len(log['execution_platform']['GPU'])
        gpu_name = processor_shortforms[log['execution_platform']['GPU']['0']['Name']]
        name = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
    else:
        name = processor_shortforms[log['execution_platform']['Processor']]
    return name


def extract_software(log):
    with open('meta_environment.json', 'r') as meta:
        ml_backends = json.load(meta)['ml_backends']
    if 'backend' in log['config']:
        backend_name = log['config']['backend']
    else:
        backend_name = list(ml_backends.keys())[0]
    backend_meta = ml_backends[backend_name]
    backend_version = 'n.a.'
    for package in backend_meta["Packages"]:
        for req in log['requirements']:
            if req.split('==')[0].replace('-', '_') == package.replace('-', '_'):
                backend_version = req.split('==')[1]
                break
        else:
            continue
        break
    return f'{backend_name} {backend_version}'
