#!/bin/bash

for filename in ./raw_videos/*.mp4; do
    base=$(basename $filename)
    output_path="./fixed_videos/$base"
    ./convert.sh $filename $output_path
done
