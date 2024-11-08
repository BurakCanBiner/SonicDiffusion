#!/bin/bash


edit_root_path=""
declare -a StringArray=( 
        "edited_image_path_1"
        "edited_image_path_2"
                )

for val in "${StringArray[@]}"; do
    echo "$val"

    python3 evaluation_publish.py --edited_images_path "${edit_root_path}${val}" \
    --dataset_image_path /datasets/audio-image/images --dataset_audio_path /datasets/audio-image/audios \
    --cfg_path inference_configs/feature-extraction.yaml \
    --labels_path dataset_files/labels.csv \
    --extension "*.png" \
    --dataset_type "landscape"  --iis --ais --wav2clip --aic

done