# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import csv
import datetime
import os
import shutil
import time

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from algos import build_algo
from datasets import build_loader
from models import build_diffusion, build_model
from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import get_degreadation_image
from utils.distributed import common_init, get_logger, init_processes
from utils.functions import get_timesteps, postprocess, preprocess, strfdt
from utils.save import save_result

torch.set_printoptions(sci_mode=False)


def main(cfg):
    print("cfg.exp.seed", cfg.exp.seed)
    common_init(dist.get_rank(), seed=cfg.exp.seed)
    torch.cuda.set_device(dist.get_rank())

    # Setup
    logger = get_logger(name="main", cfg=cfg)
    logger.info(f"Experiment name is {cfg.exp.name}")
    exp_root = cfg.exp.root
    samples_root = cfg.exp.samples_root
    exp_name = cfg.exp.name
    res_file_path = os.path.join(exp_root, samples_root, cfg.exp.res_file_name)
    samples_root = os.path.join(exp_root, samples_root, exp_name)
    dataset_name = cfg.dataset.name
    if dist.get_rank() == 0:
        if cfg.exp.overwrite:
            if os.path.exists(samples_root):
                shutil.rmtree(samples_root)
            os.makedirs(samples_root)
        else:
            if not os.path.exists(samples_root):
                os.makedirs(samples_root)

    # Build dataset
    loader = build_loader(cfg)
    logger.info(f"Dataset size is {len(loader.dataset)}")

    # Build model
    model, classifier = build_model(cfg)
    model.eval()
    if classifier is not None:
        classifier.eval()
    diffusion = build_diffusion(cfg)
    cg_model = ClassifierGuidanceModel(model, classifier, diffusion, cfg)

    # Build sampler
    algo = build_algo(cg_model, cfg)
    # TODO: Populate this dynamically
    cond_samplers = set(
        [
            "ddrm",
            "mcg",
            "dps",
            "pgdm",
            "reddiff",
            "pgdm_cont",
            "cpgdm",
            "ncpgdm",
        ]
    )
    if cfg.algo.name in cond_samplers:
        H = algo.H

    # Set up metrics
    psnr = torchmetrics.image.PeakSignalNoiseRatio()
    lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity()
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
    ms_ssim = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure()

    start_time = time.time()
    for it, (x, y, info) in enumerate(loader):
        if cfg.exp.smoke_test > 0 and it >= cfg.exp.smoke_test:
            break

        x, y = x.cuda(), y.cuda()

        # Preprocess
        x = preprocess(x)

        # Timestep discretization
        ts = get_timesteps(cfg)

        kwargs = info
        if cfg.algo.name in cond_samplers:
            idx = info["index"]
            if "inp" in cfg.algo.deg or "in2" in cfg.algo.deg:
                H.set_indices(idx)
            y_0 = H.H(x)

            # This is to account for scaling to [-1, 1]
            y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_y * 2
            kwargs["y_0"] = y_0

        # Sample
        if cfg.exp.save_evolution:
            xt_s, _, xt_vis, _, mu_fft_abs_s, mu_fft_ang_s = algo.sample(
                x, y, ts, **kwargs
            )
        else:
            xt_s, _ = algo.sample(x, y, ts, **kwargs)

        # Update metrics
        isnan = torch.isnan(xt_s[0])

        # Dont compute metrics if nan was generated
        if not (torch.sum(isnan) > 0):
            psnr.update(xt_s[0], x.cpu())
            lpips.update(
                torch.clip(xt_s[0].to(torch.float32), -1.0, 1.0),
                torch.clip(x.to(torch.float32), -1.0, 1.0).cpu(),
            )
            ssim.update(xt_s[0], x.cpu())
            ms_ssim.update(xt_s[0], x.cpu())

        # Dont save if nan was generated
        if not (torch.sum(isnan) > 0):
            if isinstance(xt_s, list):
                xo = postprocess(xt_s[0]).cpu()
            else:
                xo = postprocess(xt_s).cpu()

            save_result(dataset_name, xo, y, info, samples_root, "")

        if (cfg.algo.name in cond_samplers) and cfg.exp.save_deg:
            xo = postprocess(get_degreadation_image(y_0, H, cfg))
            save_result(dataset_name, xo, y, info, samples_root, "deg")
            # b, c, h, w = x.shape
            # save_result(dataset_name, postprocess(y_0).view(b, c, h//4, w//4), y, info, samples_root, "y")

        if cfg.exp.save_ori:
            xo = postprocess(x)
            save_result(dataset_name, xo, y, info, samples_root, "ori")

        if it % cfg.exp.logfreq == 0 or cfg.exp.smoke_test > 0 or it < 10:
            now = time.time() - start_time
            now_in_hours = strfdt(datetime.timedelta(seconds=now))
            future = (len(loader) - it - 1) / (it + 1) * now
            future_in_hours = strfdt(datetime.timedelta(seconds=future))
            logger.info(
                f"Iter {it}: {now_in_hours} has passed, expect to finish in {future_in_hours}"
            )

    # Log metrics
    if len(loader) > 0 and (cfg.algo.name in cond_samplers):
        res_dict = {
            "EXP_NAME": exp_name,
            "PSNR": f"{psnr.compute().item():.5f}",
            "LPIPS": f"{lpips.compute().item():.5f}",
            "SSIM": f"{ssim.compute().item():.5f}",
            "MS_SSIM": f"{ms_ssim.compute().item():.5f}",
        }
        logger.info(res_dict)

        # Sync and dump results to a file
        dist.barrier()

        if dist.get_rank() == 0:
            fields = ["EXP_NAME", "PSNR", "LPIPS", "SSIM", "MS_SSIM"]
            with open(res_file_path, "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writerow(res_dict)

    logger.info("Done.")
    now = time.time() - start_time
    now_in_hours = strfdt(datetime.timedelta(seconds=now))
    logger.info(f"Total time: {now_in_hours}")


# Change the hydra config here.
# TODO: Automate this to specify from the config
@hydra.main(
    version_base="1.2", config_path="_configs", config_name="imagenet256_uncond"
)
def main_dist(cfg: DictConfig):
    cwd = HydraConfig.get().runtime.output_dir

    if cfg.dist.num_processes_per_node < 0:
        size = torch.cuda.device_count()
        cfg.dist.num_processes_per_node = size
    else:
        size = cfg.dist.num_processes_per_node
    if size > 1:
        num_proc_node = cfg.dist.num_proc_node
        num_process_per_node = cfg.dist.num_processes_per_node
        world_size = num_proc_node * num_process_per_node
        mp.spawn(
            init_processes,
            args=(world_size, main, cfg, cwd),
            nprocs=world_size,
            join=True,
        )
    else:
        init_processes(0, size, main, cfg, cwd)


if __name__ == "__main__":
    main_dist()
