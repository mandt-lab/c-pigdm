import torch
import os
import argparse
import numpy as np
import lpips
import pathlib
import torchvision
from pytorch_msssim import ms_ssim
from models.utils import create_model
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from dataset_eval import get_dataset
from torch.utils.data import DataLoader
from deg_utils.degredations import Deblurring, NaiveInpainting, SRConv, PhaseRetrievalOperator, JPEG, NeuralCompression, Colorization
from sampler import PGDM, ConjugateV2, Unconditional

def parse_img_shape(argument):
    try:
        # Split the string by commas and convert it to a tuple of integers
        return tuple(int(item) for item in argument.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError("Each item in the tuple should be an integer")

def create_args():
    parser = argparse.ArgumentParser(description='Inverse Problem Configuration')
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--label', default="cat", type=str, help='label')
    parser.add_argument('--ckpt_path', default='/extra/ucibdl1/ruihan/rectified_flow_ckpts', type=str, help='checkpoint path')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument("--out_dir", type=str, default="/home/ruihay1/projects/RectifiedFlow/conjugate_results/")
    parser.add_argument("--degradation", type=str, default="sr4")
    parser.add_argument("--w", type=float)
    parser.add_argument("--skip_t", type=float)
    parser.add_argument("--img_shape", default=(3,256,256), type=parse_img_shape, help="image_shape")
    parser.add_argument("--n_sample_steps", type=int, default=5, help="number of sample steps")
    parser.add_argument("--validation", action="store_true", help="validation mode")
    parser.add_argument("--grid_search", action="store_true", help="grid search mode")
    args = parser.parse_args()
    return args

def create_data_loader(args):
    ds = get_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return dl

def load_model(args):
    if args.dataset == 'celebahq':
        from configs.rectified_flow.celeba_hq_pytorch_rf_gaussian import get_config
        config = get_config()
        ckpt = torch.load(os.path.join(args.ckpt_path, 'celeba_hq.pth'))
    elif args.dataset == 'afhq':
        from configs.rectified_flow.afhq_cat_pytorch_rf_gaussian import get_config
        config = get_config()
        ckpt = torch.load(os.path.join(args.ckpt_path, f'afhq_{args.label}.pth'))
    elif args.dataset == 'lsunbedroom':
        from configs.rectified_flow.bedroom_rf_gaussian import get_config
        config = get_config()
        ckpt = torch.load(os.path.join(args.ckpt_path, 'lsun_bedroom.pth'))
    elif args.dataset == 'lsunchurch':
        from configs.rectified_flow.church_rf_gaussian import get_config
        config = get_config()
        ckpt = torch.load(os.path.join(args.ckpt_path, 'lsun_church.pth'))
    else:
        raise NotImplementedError
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    ema.copy_to(model.parameters())
    return model

def pdf(x, sigma=10):
    return torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2]))

def create_degradation(args):
    name = args.degradation
    img_shape = args.img_shape
    if name[:2] == "sr":
        factor = int(name[2:])

        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(args.device)
        deg = SRConv(kernel / kernel.sum(), img_shape[0], img_shape[1], args.device, stride=factor)
    elif name[:4] == "jpeg":
        deg = JPEG(int(name[4:]), img_shape, norm=True if "lsun" not in args.dataset else False)
    elif name[:3] == "ntc":
        deg = NeuralCompression(int(name[3:]), img_shape, args.device, norm=True if "lsun" not in args.dataset else False)
    elif name[:3] == "inp":
        deg = NaiveInpainting(img_shape[0], img_shape[1], 'masks/20ff1.npz', args.device)
    elif name[:5] == "phase":
        oversample=2.0
        deg = PhaseRetrievalOperator(oversample=oversample, device=args.device)
    elif name[:6] == "deblur":
        sigma = 3
        window = 61
        kernel = torch.tensor([pdf(t, sigma) for t in range(-(window-1)//2, (window-1)//2)], device=args.device)
        # kernel = torch.Tensor([pdf(-2, sigma), pdf(-1, sigma), pdf(0, sigma), pdf(1, sigma), pdf(2, sigma)]).to(args.device)
        deg = Deblurring(kernel / kernel.sum(), 3, img_shape[1], args.device)
        # deg = Deblurring(torch.Tensor([1 / 9] * 9).to(args.device), 3, img_shape[1], args.device)
    elif name[:5] == "color":
        deg = Colorization(img_shape[1], device=args.device)
    else:
        raise NotImplementedError
    return deg

def normalize(x):
    return x * 2. - 1.

def unnormalize(x):
    return (x + 1.) / 2.

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    return (20 * torch.log10(1.0 / torch.sqrt(mse)))

def main():
    args = create_args()
    pgdm_dir = os.path.join(str(args.out_dir), str(args.dataset), 'sample_' + str(args.n_sample_steps), 'pgdm', 'w_' + str(args.w), 'st_' + str(args.skip_t))
    # print(conj_dir)
    # if os.path.isdir(conj_dir):
    #     exit()
    loss_fn_alex = lpips.LPIPS(net='alex').to(args.device)
    dl = create_data_loader(args)
    model = load_model(args)
    model = model.to(args.device)
    model.eval()
    model.device = args.device
    deg = create_degradation(args)
    sampler = PGDM(model, deg, w=args.w, skip_t=args.skip_t)
    # sampler = Unconditional(model)
    lpips_val, psnr_val, ms_ssim_val, num_sample = 0, 0, 0, 0
    for batch_idx, (unormed_image, _) in enumerate(dl):
        unormed_image = unormed_image.to(args.device)
        if "lsun" in args.dataset:
            normed_image = unormed_image
        else:
            normed_image = normalize(unormed_image)
        img_dim = normed_image.shape[1:]
        y = deg.H(normed_image)
        normed_rec_img_pgdm = sampler.sample_pgdm(y, img_dim, n_steps=args.n_sample_steps)
        # normed_rec_img_pgdm = sampler.sample(unormed_image.shape, sample_steps=args.n_sample_steps)
        # print(normed_rec_img_pgdm.min(), normed_rec_img_pgdm.max(), normed_rec_img_pgdm.mean(), normed_rec_img_pgdm.std())
        recon = deg.H_pinv(y).reshape(*unormed_image.shape)
        if "lsun" in args.dataset:
            unormed_rec_img_pgdm = normed_rec_img_pgdm.clamp(0, 1)
            recon = recon.clamp(0, 1)
        else:
            unormed_rec_img_pgdm = unnormalize(normed_rec_img_pgdm).clamp(0, 1)
            recon = unnormalize(recon).clamp(0, 1)

        with torch.no_grad():
            if "lsun" in args.dataset:
                lpips_val += loss_fn_alex(normalize(normed_rec_img_pgdm), normalize(normed_image)).sum()
            else:
                lpips_val += loss_fn_alex(normed_rec_img_pgdm, normed_image).sum()
            psnr_val += psnr(unormed_rec_img_pgdm, unormed_image).sum()
            ms_ssim_val += ms_ssim(unormed_rec_img_pgdm, unormed_image, size_average=False).sum()
            num_sample += normed_image.shape[0]

        pathlib.Path(os.path.join(args.out_dir, str(args.dataset), "origin")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(pgdm_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(args.out_dir, str(args.dataset), "degraded_pinv")).mkdir(parents=True, exist_ok=True)
        
        for datum_idx in range(unormed_image.shape[0]):
            # if not os.path.isfile(os.path.join(args.out_dir, str(args.dataset), "origin", f"origin_{batch_idx}_{datum_idx}.png")):
            torchvision.utils.save_image(unormed_image.cpu()[datum_idx], os.path.join(args.out_dir, str(args.dataset), "origin", f"origin_{batch_idx}_{datum_idx}.png"))
            torchvision.utils.save_image(unormed_rec_img_pgdm.cpu()[datum_idx], os.path.join(pgdm_dir, f"pgdm_{batch_idx}_{datum_idx}.png"))
            # if not os.path.isfile(os.path.join(args.out_dir, str(args.dataset), "degraded_pinv", f"degrad_pinv_{batch_idx}_{datum_idx}.png")):
            torchvision.utils.save_image(recon.cpu()[datum_idx], os.path.join(args.out_dir, str(args.dataset), "degraded_pinv", f"degrad_pinv_{batch_idx}_{datum_idx}.png"))
            # break
        if args.grid_search:
            if batch_idx == 15:
                break
        
    with open(os.path.join(pgdm_dir, 'lpips.txt'), 'w') as f:
        f.write(f"lpips: {(lpips_val / num_sample).item()}")
    with open(os.path.join(pgdm_dir, 'psnr.txt'), 'w') as f:
        f.write(f"lpips: {(psnr_val / num_sample).item()}")
    with open(os.path.join(pgdm_dir, 'ms_ssim.txt'), 'w') as f:
        f.write(f"lpips: {(ms_ssim_val / num_sample).item()}")

if __name__ == "__main__":
    main()