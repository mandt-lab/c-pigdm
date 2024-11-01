# Benchmark DPS on the Bicubic4 task

NUM_GPUS=8
CKPT_ROOT="/home/pandeyk1/ciip_results/pretrained_chkpts/adm/"
DEG=deblur_gauss
SAMPLES_ROOT="dps_${DEG}_steps\=1000_20thMay/"
EXP_ROOT="/home/pandeyk1/ciip_results/ablations/imagenet/${DEG}"
RES_FILE_NAME="results.csv"
BATCH_SIZE=2
SEED=0
STRIDE=ddpm_uniform
SMOKE_TEST=0 #Iterate over the entire dataset by default
NUM_STEPS=1000
GRAD_TERM_WEIGHT=1.0
START_TIME=0
EXP_NAME="dps__${DEG}__${START_TIME}__${NUM_STEPS}"

python main.py \
        diffusion=ddpm \
        algo=dps \
        algo.deg=$DEG \
        algo.grad_term_weight=$GRAD_TERM_WEIGHT \
        loader=imagenet256_ddrmpp \
        loader.batch_size=$BATCH_SIZE \
        loader.num_workers=2 \
        dist.num_processes_per_node=$NUM_GPUS \
        exp.name=$EXP_NAME \
        exp.num_steps=$NUM_STEPS \
        exp.seed=$SEED \
        exp.stride=$STRIDE \
        exp.t_start=$START_TIME \
        exp.t_end=1000 \
        exp.root=$EXP_ROOT \
        exp.ckpt_root=$CKPT_ROOT \
        exp.samples_root=$SAMPLES_ROOT \
        exp.res_file_name=$RES_FILE_NAME \
        exp.overwrite=True \
        exp.save_ori=False \
        exp.save_deg=False \
        exp.smoke_test=$SMOKE_TEST