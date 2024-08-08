import inspect
import os
from itertools import product

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

KERAS_BUILTINS = [e for e in tf.keras.applications.__dict__.values() if inspect.ismodule(e) and hasattr(e, 'preprocess_input')]
KERAS_MODELS = {n: e for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}
KERAS_PREPR = {n: mod.preprocess_input for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}

INCEPTION_INPUT = {
    mname: (299, 299) for mname in KERAS_MODELS if 'ception' in mname
}
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
NASNET_INPUT = {
    'NASNetLarge': (331, 331)
}
MODEL_CUSTOM_INPUT = {**INCEPTION_INPUT, **EFFICIENT_INPUT, **NASNET_INPUT}

GPU_SIZES = {
    "ConvNeXtBase": 64,
    "ConvNeXtLarge": 64,
    "ConvNeXtSmall": 32,
    "ConvNeXtTiny": 32,
    "ConvNeXtXLarge": 64,
    "DenseNet121": 32,
    "DenseNet169": 64,
    "DenseNet201": 64,
    "EfficientNetB0": 32,
    "EfficientNetB1": 64,
    "EfficientNetB2": 64,
    "EfficientNetB3": 32,
    "EfficientNetB4": 64,
    "EfficientNetB5": 64,
    "EfficientNetB6": 32,
    "EfficientNetB7": 64,
    "EfficientNetV2B0": 32,
    "EfficientNetV2B1": 64,
    "EfficientNetV2B2": 64,
    "EfficientNetV2B3": 64,
    "InceptionResNetV2": 64,
    "InceptionV3": 32,
    "MobileNet": 64,
    "MobileNetV2": 64,
    "NASNetLarge": 64,
    "NASNetMobile": 32,
    "ResNet50": 32,
    "ResNet101": 64,
    "ResNet152": 64,
    "ResNet50V2": 64,
    "ResNet101V2": 32,
    "ResNet152V2": 64,
    "VGG16": 64,
    "VGG19": 64,
    "Xception": 64,
    "EfficientNetV2L": 64,
    "EfficientNetV2M": 64,
    "EfficientNetV2S": 64,
}

CPU_SIZES = {
    "ConvNeXtBase": 64,
    "ConvNeXtLarge": 64,
    "ConvNeXtSmall": 64,
    "ConvNeXtTiny": 64,
    "ConvNeXtXLarge": 64,
    "DenseNet121": 64,
    "DenseNet169": 64,
    "DenseNet201": 64,
    "EfficientNetB0": 64,
    "EfficientNetB1": 64,
    "EfficientNetB2": 64,
    "EfficientNetB3": 64,
    "EfficientNetB4": 64,
    "EfficientNetB5": 64,
    "EfficientNetB6": 64,
    "EfficientNetB7": 64,
    "EfficientNetV2B0": 64,
    "EfficientNetV2B1": 64,
    "EfficientNetV2B2": 64,
    "EfficientNetV2B3": 64,
    "EfficientNetV2L": 64,
    "EfficientNetV2M": 64,
    "EfficientNetV2S": 64,
    "InceptionResNetV2": 64,
    "InceptionV3": 64,
    "MobileNet": 64,
    "MobileNetV2": 64,
    "NASNetLarge": 64,
    "NASNetMobile": 64,
    "ResNet50": 64,
    "ResNet101": 64,
    "ResNet152": 64,
    "ResNet50V2": 64,
    "ResNet101V2": 64,
    "ResNet152V2": 64,
    "VGG16": 64,
    "VGG19": 64,
    "Xception": 64,
}

CORRUPTIONS = [
    'gaussian_noise_',
    'shot_noise_',
    'impulse_noise_',
    'defocus_blur_',
    'glass_blur_',
    'motion_blur_',
    'zoom_blur_',
    'snow_',
    'frost_',
    'fog_',
    'brightness_',
    'contrast_',
    'elastic_transform_',
    'pixelate_',
    'jpeg_compression_',
    'gaussian_blur_',
    'saturate_',
    'spatter_',
    'speckle_noise_'
]
ALL_CORRUPTIONS = [f'imagenet2012_corrupted/{corr}{level}' for corr, level in product(CORRUPTIONS, [1, 2, 3, 4, 5])]
FIRST_CORR = ALL_CORRUPTIONS[:10]

def load_corrupted_sample(data_path, seed=0):
    # rng = np.random.default_rng(0)
    # IDX = tf.data.Dataset.range(50000)
    complete = []
    for idx, corr in enumerate(FIRST_CORR):
        _, ds, info = load_data_and_model(data_path, variant=corr, batch_size=1)
        shard = ds.shard(len(FIRST_CORR), index=idx)
        complete.append(shard)
        # idc = IDX.shard(len(FIRST_CORR), index=idx)
        # for img, i in tqdm( zip(shard, idc), total=len(list(idc)) ):  
        #     im = Image.fromarray(img[0].numpy())
        #     im.save(f"imgs/{str(i.numpy()).zfill(5)}_img_{corr.split('/')[1]}.jpeg")

    # for idx, img in enumerate(all):
    #     print(idx)
    # print(1)
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

def load_data_and_model(data_path, model=None, variant='imagenet2012', batch_size=32):
    # data
    extract_dir = os.path.join(data_path, 'extracted')
    config = {'download_config': tfds.download.DownloadConfig(extract_dir=extract_dir, manual_dir=data_path)}
    if variant is 'corrupted_sample':
        ds, info = load_corrupted_sample(data_path)
    else:
        ds, info = tfds.load(variant, data_dir=extract_dir, split='validation', download=True, shuffle_files=False, as_supervised=True, with_info=True, download_and_prepare_kwargs=config)
    if model is None:
        return None, ds, info
    preprocessor = load_prepr(model)
    ds = ds.map(preprocessor)
    ds = ds.batch(batch_size)
    # model
    model = KERAS_MODELS[model](weights='imagenet')
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD()
    metrics = ['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)
    return model, ds, info
