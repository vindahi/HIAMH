###!/bin/bash
set -e
METHOD='FSFH'
bits=(128)

for i in ${bits[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=0 python train.py --nbit $i \
                                                     --dataset flickr \
                                                     --epochs 140 \
                                                     --dropout 0.1 \
                                                     --mlpdrop 0.1 \

  echo "**********End ${METHOD} algorithm**********"
done


# set -e
# METHOD='FSFH'
# bits=(128)
# paralist=(0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000)

# for i in ${bits[*]}; do
# for j in ${paralist[*]}; do
#   echo "**********Start ${METHOD} algorithm**********"
#   CUDA_VISIBLE_DEVICES=0 python train.py --nbit $i \
#                                                      --dataset flickr \
#                                                      --epochs 140 \
#                                                      --dropout 0.1 \
#                                                      --mlpdrop 0.1 \
#                                                      --param_clf $j

#   echo "**********End ${METHOD} algorithm**********"
# done
# done


