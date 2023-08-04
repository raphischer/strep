import json


def calc_accuracy(res, train=False, metric='accuracy'):
    split = 'train' if train else 'validation'
    return res[split]['results']['metrics'][metric] * 100


def calc_parameters(res):
    if 'validation' in res:
        return res['validation']['results']['model']['params']
    return res['results']['model']['params']


def calc_flops(res):
    if 'validation' in res:
        return res['validation']['results']['model']['flops']
    return res['results']['model']['flops']


def calc_fsize(res):
    if 'validation' in res:
        return res['validation']['results']['model']['fsize']
    return res['results']['model']['fsize']


def calc_inf_time(res):
    return res['validation']['duration'] # TODO / 50000


def calc_power_draw(res):
    power_draw = 0
    if res['validation']["monitoring_pynvml"] is not None:
        power_draw += res['validation']["monitoring_pynvml"]["total"]["total_power_draw"]
    if res['validation']["monitoring_pyrapl"] is not None:
        power_draw += res['validation']["monitoring_pyrapl"]["total"]["total_power_draw"]
    return power_draw


def calc_time_train(res, per_epoch=False):
    if len(res['results']['history']) < 1:
        if per_epoch:
            return None
        else:
            return res["duration"]
    val_per_epoch = res["duration"] / len(res["results"]["history"]["loss"])
    return val_per_epoch


def calc_power_draw_train(res, per_epoch=False):
    power_draw = 0
    if res["monitoring_pynvml"] is not None:
        power_draw += res["monitoring_pynvml"]["total"]["total_power_draw"]
    if res["monitoring_pyrapl"] is not None:
        power_draw += res["monitoring_pyrapl"]["total"]["total_power_draw"]
    # if there is no information on training epochs
    if len(res['results']['history']) < 1:
        if per_epoch:
            return None
        else:
            return power_draw
    val_per_epoch = power_draw / len(res["results"]["history"]["loss"])
    return val_per_epoch


# def characterize_monitoring(summary):
#     sources = {
#         'GPU': ['NVML'] if summary['monitoring_pynvml'] is not None else [],
#         'CPU': ['RAPL'] if summary['monitoring_pyrapl'] is not None else [],
#         'Extern': []
#     }
#     # TODO also make use of summary['monitoring_psutil']
#     # if summary['monitoring_psutil'] is not None:
#     #     sources['CPU'].append('psutil')
#     return sources


def extract_architecture(log):
    try:
        with open('meta_environment.json', 'r') as meta:
            processor_shortforms = json.load(meta)['processor_shortforms']
        if 'GPU' in log['execution_platform'] and len(log['execution_platform']['GPU']) > 0:
            n_gpus = len(log['execution_platform']['GPU'])
            gpu_name = processor_shortforms[log['execution_platform']['GPU']['0']['Name']]
            name = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
        else:
            name = processor_shortforms[log['execution_platform']['Processor']]
        return name
    except Exception:
        return 'n.a.'


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


PROPERTIES = {
    'meta': {
        'task': lambda log: log['directory_name'].split('_')[0],
        'dataset': lambda log: log['config']['dataset'],
        'model': lambda log: log['config']['model'],
        'architecture': lambda log: extract_architecture(log),
        'software': lambda log: extract_software(log),
    },

    'train': {
        'train_running_time': lambda model_log: calc_time_train(model_log),
        'train_power_draw': lambda model_log: calc_power_draw_train(model_log)
    },
    
    'infer': {
        'running_time': lambda model_log: calc_inf_time(model_log),
        'power_draw': lambda model_log: calc_power_draw(model_log),
        'top-1_val': lambda model_log: calc_accuracy(model_log),
        'top-5_val': lambda model_log: calc_accuracy(model_log, metric='top_5_accuracy'),
        'f1_val': lambda model_log: calc_accuracy(model_log, metric='f1'),
        'precisision_val': lambda model_log: calc_accuracy(model_log, metric='precision'),
        'recall_val': lambda model_log: calc_accuracy(model_log, metric='recall'),
        'parameters': lambda model_log: calc_parameters(model_log),
        'fsize': lambda model_log: calc_fsize(model_log),
        'flops': lambda model_log: calc_flops(model_log, )
    }
}
