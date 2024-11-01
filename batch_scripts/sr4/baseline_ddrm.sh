# Benchmark DDRM on the Bicubic4 task

NUM_GPUS=7
CKPT_ROOT="/home/pandeyk1/ciip_results/pretrained_chkpts/adm/"
DEG=deblur_gauss
SAMPLES_ROOT="ddrm_${DEG}_steps\=20_17thMay/"
EXP_ROOT="/home/pandeyk1/ciip_results/ablations/imagenet/${DEG}"
RES_FILE_NAME="results.csv"
BATCH_SIZE=2
SEED=0
STRIDE=ddpm_uniform
SMOKE_TEST=0 #Iterate over the entire dataset by default
NUM_STEPS=20
ETA=0.85
ETA_B=1.0
START_TIME=0
EXP_NAME="ddrm__${DEG}__${START_TIME}__${NUM_STEPS}"

python main.py \
        diffusion=ddpm \
        algo=ddrm \
        algo.deg=$DEG \
        algo.eta=$ETA \
        algo.eta_b=$ETA_B \
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