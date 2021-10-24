#!/bin/bash

input_file=$1
output_file=$2

# tareeka 1
# this increases video rate but framerate is unknown?
# old rate / new rate is 25 / 15 = 

framerate=$(ffmpeg -i $input_file 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")
pts_coeff=$(echo "$framerate / 15.0" | bc -l)
rounded=$(printf %.2f $pts_coeff)
ffmpeg -y -i $input_file -vf "setpts=$rounded*PTS" -r 15 temp.mp4

# tareeka 2
# this does not increase the duration of video
# ffmpeg -i $input_file -filter:v fps=fps=15 temp.mp4

ffmpeg -i temp.mp4 -vf scale=640:480 $output_file
rm temp.mp4
