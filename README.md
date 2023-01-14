# Facelapse

A set of scripts for making a timelapse of a person from a series of photographs.

## Goals

- Take a folder of photographs
- Line up their pupils
- Stitch them into a video file

## Status

The lining up isn't perfect, but it kinda works!

## Usage

python src/face-lapse.py -h
usage: face-lapse.py [-h] -i INPUT_FOLDER -c OPENCV_FOLDER -v VIDEO_OUTPUT
                     [-f FRAMES_PER_SECOND] [-o OUTPUT_FOLDER]

options:
  -h, --help            show this help message and exit
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Path to the folder containing the face images
  -c OPENCV_FOLDER, --opencv_folder OPENCV_FOLDER
                        Path to the folder where opencv is installed.
  -v VIDEO_OUTPUT, --video_output VIDEO_OUTPUT
                        Path to the output video file
  -f FRAMES_PER_SECOND, --frames_per_second FRAMES_PER_SECOND
                        Number of seconds per picture in the output video
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to save processed files in
