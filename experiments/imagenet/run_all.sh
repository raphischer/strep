#! /bin/bash

# Get current timestamp for experiment naming
printf -v now '%(%F_%H-%M-%S)T' -1
exp_name="imagenet_$1_$now"
exp_create_str=$(mlflow experiments create -n $exp_name)
echo $exp_create_str
exp_id=$(echo $exp_create_str | awk '{print $NF}')

# Define model architectures to iterate over
models=("ConvNeXtBase" "ConvNeXtSmall" "ConvNeXtTiny" "DenseNet121" "DenseNet169" "DenseNet201" "EfficientNetB0" 
        "EfficientNetB1" "EfficientNetB2" "EfficientNetB3" "EfficientNetB4" "EfficientNetB5" "EfficientNetB6" 
        "EfficientNetB7" "EfficientNetV2B0" "EfficientNetV2B1" "EfficientNetV2B2" "EfficientNetV2B3" "EfficientNetV2L" 
        "EfficientNetV2M" "EfficientNetV2S" "InceptionResNetV2" "InceptionV3" "MobileNet" "MobileNetV2" 
        "NASNetLarge" "NASNetMobile" "ResNet50" "ResNet101" "ResNet152" "ResNet50V2" "ResNet101V2" "ResNet152V2" 
        "Xception" "MobileNetV3Large" "MobileNetV3Small") # VGG16 VGG19 "ConvNeXtLarge" "ConvNeXtXLarge"

# Loop over all models and GPUs
for m in "${models[@]}"
do
    for g in "0" "1"
    do
        # Keep trying the mlflow run until it succeeds
        while true
        do
            echo "Running model $m on GPU $g ..."
            mlflow run --experiment-name=$exp_name -e main.py -P model=$m -P datadir=/data/d1/fischer_diss/imagenet -P seconds=$1 -P nogpu=$g ./experiments/imagenet
            
            # Check if the mlflow run succeeded (exit status 0)
            if [ $? -eq 0 ]; then
                echo "Run succeeded for model $m on GPU $g"
                break  # Exit the loop if succeeded
            else
                echo "Run failed for model $m on GPU $g, retrying..."
            fi
        done
    done

    # Save experiment data to CSV
    mlflow experiments csv -x $exp_id > "$exp_name.csv"
done