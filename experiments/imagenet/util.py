import inspect
import os

import tensorflow as tf
import tensorflow_datasets as tfds

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

def load_prepr(model_name):
    prepr = KERAS_PREPR[model_name]
    input_size = MODEL_CUSTOM_INPUT.get(model_name, (224, 224))
    return lambda img, label : preprocess(img, label, prepr, input_size)

def preprocess(image, label, prepr_func, input_size):
    i = tf.cast(image, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, input_size[0], input_size[1]) # necessary for processing batches
    i = prepr_func(i)
    return (i, label)

def load_data_and_model(data_path, model, corrupted=False, batch_size=32):
    # data
    extract_dir = os.path.join(data_path, 'extracted')
    config = {'download_config': tfds.download.DownloadConfig(extract_dir=extract_dir, manual_dir=data_path)}
    variant = 'imagenet2012_corrupted' if corrupted else 'imagenet2012'
    ds, info = tfds.load(variant, data_dir=extract_dir, split='validation', download=True, shuffle_files=False, as_supervised=True, with_info=True, download_and_prepare_kwargs=config)
    preprocessor = load_prepr(model)
    ds = ds.map(preprocessor)
    ds = ds.batch(batch_size)
    # model
    model = KERAS_MODELS[model](weights='imagenet')
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD()
    metrics = ['sparse_categorical_crossentropy', 'sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)
    return model, ds, info