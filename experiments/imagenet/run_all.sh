for m in "ConvNeXtBase" "ConvNeXtLarge" "ConvNeXtSmall" "ConvNeXtTiny" "ConvNeXtXLarge" "DenseNet121" "DenseNet169" "DenseNet201" "EfficientNetB0" "EfficientNetB1" "EfficientNetB2" "EfficientNetB3" "EfficientNetB4" "EfficientNetB5" "EfficientNetB6" "EfficientNetB7" "EfficientNetV2B0" "EfficientNetV2B1" "EfficientNetV2B2" "EfficientNetV2B3" "EfficientNetV2L" "EfficientNetV2M" "EfficientNetV2S" "InceptionResNetV2" "InceptionV3" "MobileNet" "MobileNetV2" "NASNetLarge" "NASNetMobile" "ResNet50" "ResNet101" "ResNet152" "ResNet50V2" "ResNet101V2" "ResNet152V2" "VGG16" "VGG19" "Xception"
do
    mlflow run -e main.py -P model=$m -P datadir=/data/d1/fischer_diss/imagenet ./experiments/imagenet
done

for m in "ConvNeXtBase" "ConvNeXtLarge" "ConvNeXtSmall" "ConvNeXtTiny" "ConvNeXtXLarge" "DenseNet121" "DenseNet169" "DenseNet201" "EfficientNetB0" "EfficientNetB1" "EfficientNetB2" "EfficientNetB3" "EfficientNetB4" "EfficientNetB5" "EfficientNetB6" "EfficientNetB7" "EfficientNetV2B0" "EfficientNetV2B1" "EfficientNetV2B2" "EfficientNetV2B3" "EfficientNetV2L" "EfficientNetV2M" "EfficientNetV2S" "InceptionResNetV2" "InceptionV3" "MobileNet" "MobileNetV2" "NASNetLarge" "NASNetMobile" "ResNet50" "ResNet101" "ResNet152" "ResNet50V2" "ResNet101V2" "ResNet152V2" "VGG16" "VGG19" "Xception"
do
    mlflow run -e main.py -P model=$m -P datadir=/data/d1/fischer_diss/imagenet -P nogpu=1 ./experiments/imagenet
done