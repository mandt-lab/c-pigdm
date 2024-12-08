samples_root=demo_samples/
exp_root=/home/pandeyk1/ciip_results/
ckpt_root=/home/pandeyk1/ciip_results/pretrained_chkpts/adm/
save_deg=True
save_ori=True
overwrite=True
smoke_test=1 # Controls the number of batches generated
batch_size=1


python main.py \
        diffusion=vpsde \
        classifier=none \
        algo=cpgdm \
        algo.lam=0 \
        algo.w=15.0 \
        algo.deg=bicubic4 \
        algo.num_eps=1e-6 \
        algo.denoise=False \
        loader=imagenet256_ddrmpp \
        loader.batch_size=$batch_size \
        loader.num_workers=2 \
        dist.num_processes_per_node=1 \
        exp.name=debug \
        exp.num_steps=5 \
        exp.seed=0 \
        exp.stride=uniform \
        exp.t_start=0.6 \
        exp.t_end=0.999 \
        exp.root=$exp_root \
        exp.name=demo \
        exp.ckpt_root=$ckpt_root \
        exp.samples_root=$samples_root \
        exp.overwrite=True \
        exp.save_ori=$save_ori \
        exp.save_deg=$save_deg \
        exp.smoke_test=$smoke_test
