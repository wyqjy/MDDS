#!/bin/bash

DATASET=$1
seed=$2


if [ ${DATASET} == pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == officehome ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
fi

for SEED in $(seq 0 0)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        python train.py \
        --source ${S1} ${S2} ${S3} \
        --target ${T} \
        --dataset ${DATASET} \
        --seed ${seed}
    done
done