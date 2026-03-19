#!/bin/sh

# This script follows the QoMEX 2026 Grand Challenge CLI requirements.
# $1: {video_folder}  - Local path to the folder containing MP4s
# $2: {pvs_video}     - Filename of the distorted video
# $3: {ref_video}     - Filename of the reference video (passed as 'x' for NR)
# $4: {report_folder} - Local path where you want the output.txt saved
# $5: {result_file}    - Name of the output file (e.g., output.txt)
# $6: {tmp_folder}     - Local path for temporary processing
# $7: {image_name}     - The name of your built docker image (vqm-test)

docker run --rm --gpus all \
  -v "$1":/data/videos \
  -v "$4":/data/reports \
  -v "$6":/data/tmp \
  "$7" /data/videos/"$2" /data/videos/"$3" /data/reports/"$5"
