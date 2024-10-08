import inspect
import os
from itertools import product
import platform
import re
import subprocess
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

KERAS_BUILTINS = [e for e in tf.keras.applications.__dict__.values() if inspect.ismodule(e) and hasattr(e, 'preprocess_input')]
KERAS_MODELS = {n: e for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}
KERAS_MODELS['MobileNetV3Large'] = tf.keras.applications.MobileNetV3Large
KERAS_MODELS['MobileNetV3Small'] = tf.keras.applications.MobileNetV3Small
KERAS_PREPR = {n: mod.preprocess_input for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}
KERAS_PREPR['MobileNetV3Large'] = tf.keras.applications.mobilenet_v3.preprocess_input
KERAS_PREPR['MobileNetV3Small'] = tf.keras.applications.mobilenet_v3.preprocess_input

EFFICIENT_INPUT = {
    'EfficientNetB1': (240, 240),
    'EfficientNetB2': (260, 260),
    'EfficientNetB3': (300, 300),
    'EfficientNetB4': (380, 380),
    'EfficientNetB5': (456, 456),
    'EfficientNetB6': (528, 528),
    'EfficientNetB7': (600, 600),
    'EfficientNetV2B1': (240, 240),
    'EfficientNetV2B2': (260, 260),
    'EfficientNetV2B3': (300, 300),
    'EfficientNetV2B4': (380, 380),
    'EfficientNetV2B5': (456, 456),
    'EfficientNetV2B6': (528, 528),
    'EfficientNetV2B7': (600, 600),
    'EfficientNetV2L': (480, 480),
    'EfficientNetV2M': (480, 480),
    'EfficientNetV2S': (384, 384)
}
NASNET_INPUT = { 'NASNetLarge': (331, 331) }
INCEPTION_INPUT = { mname: (299, 299) for mname in KERAS_MODELS if 'ception' in mname }
MODEL_CUSTOM_INPUT = {**INCEPTION_INPUT, **EFFICIENT_INPUT, **NASNET_INPUT}

CORRUPTIONS = [
    'gaussian_noise_', 'shot_noise_', 'impulse_noise_', 'defocus_blur_', 'glass_blur_', 'motion_blur_',
    'zoom_blur_', 'snow_', 'frost_', 'fog_', 'brightness_', 'contrast_', 'elastic_transform_',
    'pixelate_', 'jpeg_compression_', 'gaussian_blur_', 'saturate_', 'spatter_', 'speckle_noise_'
]
ALL_CORRUPTIONS = [f'imagenet2012_corrupted/{corr}{level}' for corr, level in product(CORRUPTIONS, [1, 2, 3, 4, 5])]
FIRST_CORR = ALL_CORRUPTIONS[:10]

TESTED_BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]

def load_corrupted_sample(data_path, seed=0):
    complete = []
    for idx, corr in enumerate(FIRST_CORR):
        _, ds, info = load_data_and_model(data_path, variant=corr, batch_size=1)
        shard = ds.shard(len(FIRST_CORR), index=idx)
        complete.append(shard)
    all = tf.data.Dataset.sample_from_datasets(complete)
    return all, info

def load_prepr(model_name):
    prepr = KERAS_PREPR[model_name]
    input_size = MODEL_CUSTOM_INPUT.get(model_name, (224, 224))
    return lambda img, label : preprocess(img, label, prepr, input_size)

def preprocess(image, label, prepr_func, input_size):
    i = tf.cast(image, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, input_size[0], input_size[1]) # necessary for processing batches
    i = prepr_func(i)
    return (i, label)

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode('ascii')
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1).strip()
    return ""

def load_data_and_model(data_path, model=None, variant='imagenet2012', batch_size=-1):
    # data
    extract_dir = os.path.join(data_path, 'extracted')
    config = {'download_config': tfds.download.DownloadConfig(extract_dir=extract_dir, manual_dir=data_path)}
    if variant is 'corrupted_sample':
        ds, __builtins__ = load_corrupted_sample(data_path)
    else:
        ds, _ = tfds.load(variant, data_dir=extract_dir, split='validation', download=True, shuffle_files=False, as_supervised=True, with_info=True, download_and_prepare_kwargs=config)
    if model is None:
        return None, ds, _
    preprocessor = load_prepr(model)
    ds = ds.map(preprocessor)

    # model
    model = KERAS_MODELS[model](weights='imagenet')
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD()
    metrics = ['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

    # data batching
    if isinstance(batch_size, int) and batch_size > 0: # use given
        batched = ds.batch(batch_size)
    elif batch_size == -1: # identify optimal
        optimal, fastest = min(TESTED_BATCH_SIZES), np.inf
        for batch_size in TESTED_BATCH_SIZES:
            batched = ds.batch(batch_size)
            try:
                t0 = time.time()
                model.evaluate(batched.take(max(TESTED_BATCH_SIZES) * 5 // batch_size))
                t1 = time.time()
                print(f'{str(batch_size):<4} {t1-t0:4.3f}')
                if t1-t0 < fastest:
                    optimal = batch_size
                    fastest = t1-t0
                else:
                    break
            except Exception:
                break
        batched = ds.batch(optimal)
        batch_size = optimal
    else: # do not batch
        raise RuntimeError('Invalid batch size')
    
    # assemble meta information
    meta = {
        "batch_size": batch_size,
        "software": f'Tensorflow {tf.__version__}',
        "architecture": get_processor_name()
    }
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices: # override with GPU information
        meta["architecture"] = tf.config.experimental.get_device_details(gpu_devices[0]).get('device_name', 'Unknown GPU')
        
    return model, batched, meta
