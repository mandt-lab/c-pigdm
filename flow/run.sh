#!/bin/bash

w=20
st=0.6 # initial temerpature for flow matching sampling. Can be a list of values for grid search
lamb=0.2 # hyperparameter for B matrix. Can be a list of values for grid search
n_sample_steps=10
deg="ntc1" # degradation type. See conjugate_rf.py for more details
dataset="lsunbedroom" # dataset

out_dir="*" #output directory
ckpt_dir="*" # checkpoints directory (NOTE this is directory not the actual model weight file)

python conjugate_rf.py --batch_size 8 --n_sample_steps $n_sample_steps --w $w --skip_t $st --lamb $lamb --degradation ${deg} --dataset ${dataset} --out_dir "${out_dir}" --ckpt_path "${ckpt_dir}"
