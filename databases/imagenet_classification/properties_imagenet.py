import json


MODEL_INFO = {
    'ResNet50':          {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet101':         {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet152':         {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG16':             {'epochs': 90, 'url': 'https://arxiv.org/abs/1409.1556'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG19':             {'epochs': 90, 'url': 'https://arxiv.org/abs/1409.1556'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'EfficientNetB0':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB1':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB2':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB3':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB4':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB5':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB6':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB7':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'RegNetX400MF':      {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX32GF':       {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX8GF':        {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext50':         {'epochs': 100, 'url': 'https://arxiv.org/abs/1611.05431'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext101':        {'epochs': 100, 'url': 'https://arxiv.org/abs/1611.05431'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'DenseNet121':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'DenseNet169':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'DenseNet201':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'Xception':          {'epochs': None, 'url': 'https://arxiv.org/abs/1610.02357'}, # no information on epochs
    'InceptionResNetV2': {'epochs': 200, 'url': 'https://arxiv.org/abs/1602.07261'},
    'InceptionV3':       {'epochs': 100, 'url': 'https://arxiv.org/abs/1512.00567'},
    'NASNetMobile':      {'epochs': 100, 'url': 'https://arxiv.org/pdf/1707.07012'},
    'MobileNetV2':       {'epochs': 300, 'url': 'https://arxiv.org/abs/1801.04381'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Small':  {'epochs': 600, 'url': 'https://arxiv.org/pdf/1905.02244'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Large':  {'epochs': 600, 'url': 'https://arxiv.org/pdf/1905.02244'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'QuickNetSmall':     {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'}, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNet':          {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'}, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNetLarge':     {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'} # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
}


def calc_accuracy(res, train=False, top5=False):
    split = 'train' if train else 'validation'
    metric = 'top_5_accuracy' if top5 else 'accuracy'
    return res[split]['results']['metrics'][metric]


def calc_parameters(res):
    if 'validation' in res:
        return res['validation']['results']['model']['params'] * 1e-6
    return res['results']['model']['params'] * 1e-6


def calc_gflops(res):
    if 'validation' in res:
        return res['validation']['results']['model']['flops'] * 1e-9
    return res['results']['model']['flops'] * 1e-9


def calc_fsize(res):
    if 'validation' in res:
        return res['validation']['results']['model']['fsize'] * 1e-6
    return res['results']['model']['fsize'] * 1e-6


def calc_inf_time(res):
    return res['validation']['duration'] / 50000 * 1000


def calc_power_draw(res):
    # TODO add the RAPL measurements if available
    power_draw = 0
    if res['validation']["monitoring_pynvml"] is not None:
        power_draw += res['validation']["monitoring_pynvml"]["total"]["total_power_draw"]
    if res['validation']["monitoring_pyrapl"] is not None:
        power_draw += res['validation']["monitoring_pyrapl"]["total"]["total_power_draw"]
    return power_draw / 50000


def calc_power_draw_train(res, per_epoch=False):
    # TODO add the RAPL measurements if available
    val_per_epoch = res["monitoring_pynvml"]["total"]["total_power_draw"] / len(res["results"]["history"]["loss"])
    val_per_epoch /= 3600000 # Ws to kWh
    if not per_epoch:
        if MODEL_INFO[res["config"]["model"]]['epochs']:
            val_per_epoch *= MODEL_INFO[res["config"]["model"]]['epochs']
        else:
            val_per_epoch = None
    return val_per_epoch


def calc_time_train(res, per_epoch=False):
    val_per_epoch = res["duration"] / len(res["results"]["history"]["loss"])
    val_per_epoch /= 3600 # s to h
    if not per_epoch:
        if MODEL_INFO[res["config"]["model"]]['epochs']:
            val_per_epoch *= MODEL_INFO[res["config"]["model"]]['epochs']
        else:
            val_per_epoch = None
    return val_per_epoch


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
    if 'name' in backend_meta:
        backend_name = backend_meta['name']
    backend_version = 'n.a.'
    for package in backend_meta["packages"]:
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
        'dataset': lambda log: 'imagenet',
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
        'top-5_val': lambda model_log: calc_accuracy(model_log, top5=True),
        'parameters': lambda model_log: calc_parameters(model_log),
        'fsize': lambda model_log: calc_fsize(model_log),
        'flops': lambda model_log: calc_gflops(model_log)
    }
}