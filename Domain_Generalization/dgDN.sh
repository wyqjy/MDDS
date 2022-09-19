#!/bin/bash

DATASET=$1
seed=$2


D1=quickdraw
D2=sketch
D3=real
D4=infograph
D5=painting
D6=clipart

for SEED in $(seq 0 0)
do
    for SETUP in $(seq 1 6)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            S4=${D5}
            S5=${D6}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            S4=${D5}
            S5=${D6}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            S4=${D5}
            S5=${D6}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            S4=${D5}
            S5=${D6}
            T=${D4}
        elif [ ${SETUP} == 5 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            S4=${D4}
            S5=${D6}
            T=${D5}
        elif [ ${SETUP} == 6 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            S4=${D4}
            S5=${D5}
            T=${D6}
        fi

        python train.py \
        --source ${S1} ${S2} ${S3} ${S4} ${S5} \
        --target ${T} \
        --dataset ${DATASET} \
        --seed ${seed}
    done
done