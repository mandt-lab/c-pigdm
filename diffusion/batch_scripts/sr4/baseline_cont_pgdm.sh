# Used for running continuous time pigdm on superres (4x bicubic)

NUM_GPUS=7
CKPT_ROOT="/home/pandeyk1/ciip_results/pretrained_chkpts/adm/"
DEG=deblur_gauss
SAMPLES_ROOT="cont_pgdm_${DEG}_st_w_17thMay/"
EXP_ROOT="/home/pandeyk1/ciip_results/ablations/imagenet/${DEG}"
RES_FILE_NAME="results.csv"
BATCH_SIZE=4
SEED=0
STRIDE=uniform
LAM=0
SMOKE_TEST=0 #Iterate over the entire dataset by default

# ablation variables

num_steps=(20)
start_time=(0.6)
# w=$(seq 3 8)
w=(6)

for START_TIME in ${start_time[@]}; do
for NUM_STEPS in ${num_steps[@]}; do
for W in $w; do

EXP_NAME="${START_TIME}__${NUM_STEPS}__${W}"

echo $EXP_NAME

python main.py \
        diffusion=vpsde \
        algo=pgdm_ddim \
        algo.lam=$LAM \
        algo.w=$W \
        algo.deg=$DEG \
        algo.sigma_y=0 \
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
        exp.save_deg=False \
        exp.smoke_test=$SMOKE_TEST

done
done
done