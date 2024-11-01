# Used for ablation studies on the Bicubic 4 (i.e. 4x super-resolution task) 
# on the imagenet validation-1k dataset (sr4-top1k.txt)
# Sampler used: Improved PGDM

NUM_GPUS=8
CKPT_ROOT="/home/pandeyk1/ciip_results/pretrained_chkpts/adm/"
DEG=bicubic4
DATASET=imagenet
SIGY=0.05
SAMPLES_ROOT=\"ipgdm_${DEG}_noisy_sigy=${SIGY}_21thMay/\"
EXP_ROOT="/home/pandeyk1/ciip_results/ablations/${DATASET}/${DEG}"
RES_FILE_NAME="results.csv"
BATCH_SIZE=4
SEED=0
STRIDE=uniform
SMOKE_TEST=0 #Iterate over the entire dataset by default
START_TIME=0.6
NUM_STEPS=5
VERSION=v5
W=2
LAM=-0.15

# ablation variables

# num_steps=(5)
# w=$(seq 13 14)
# lam=(-1.0 -0.8 -0.6 -0.4 -0.2 -0.1 -0.05)

# for START_TIME in ${start_time[@]}; do
# for NUM_STEPS in ${num_steps[@]}; do
# for LAM in ${lam[@]}; do
# for W in $w; do

EXP_NAME="${START_TIME}__${NUM_STEPS}__${W}__${LAM}"

echo $EXP_NAME

python main.py \
        diffusion=vpsde \
        algo=imp_pgdm \
        algo.version=$VERSION \
        algo.lam=$LAM \
        algo.ms_order=1 \
        algo.w=$W \
        algo.deg=$DEG \
        algo.sigma_y=$SIGY \
        algo.num_eps=1e-6 \
        algo.denoise=False \
        loader=imagenet256_ddrmpp \
        loader.batch_size=$BATCH_SIZE \
        loader.num_workers=2 \
        dist.num_processes_per_node=$NUM_GPUS \
        exp.name="${EXP_NAME}" \
        exp.num_steps=$NUM_STEPS \
        exp.seed=$SEED \
        exp.stride=$STRIDE \
        exp.t_start=$START_TIME \
        exp.t_end=0.999 \
        exp.root=$EXP_ROOT \
        exp.ckpt_root=$CKPT_ROOT \
        exp.samples_root=$SAMPLES_ROOT \
        exp.res_file_name=$RES_FILE_NAME \
        exp.overwrite=True \
        exp.save_ori=False \
        exp.save_deg=True \
        exp.smoke_test=$SMOKE_TEST

# done
# done
# done