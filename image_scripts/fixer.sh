#!/bin/bash

for filename in data/submission/raw_videos/*.mp4; do
    base=$(basename $filename)
    output_path="data/submission/fixed_videos/$base"
    image_scripts/convert.sh $filename $output_path
done
