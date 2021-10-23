#!/bin/bash

input_file=$1
output_file=$2
ffmpeg -i $input_file -filter:v fps=fps=15 temp.mp4
ffmpeg -i temp.mp4 -vf scale=640:480 $output_file
rm temp.mp4
