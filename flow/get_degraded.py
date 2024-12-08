import torch
import numpy as np
import torchvision
from deg_utils.degredations import Quantization, NaiveInpainting, SRConv, PhaseRetrievalOperator, JPEG, NeuralCompression, Colorization, Deblurring

def pdf(x, sigma=10):
    return torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2]))

def main():
    img_path = "/extra/ucibdl1/shared/projects/conjugate_inverse/kp_results/ground_truth/imagenet/samples/n01514668/ILSVRC2012_val_00000329_ori.png"
    factor = int(4)
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
    kernel = torch.from_numpy(k).float().to("cuda")
    deg = SRConv(kernel / kernel.sum(), 3, 256, "cuda", stride=factor)
    img = torchvision.io.read_image(img_path).float().to("cuda").unsqueeze(0) / 255.
    sr_img = deg.H(img).reshape(3, 64, 64)
    torchvision.utils.save_image(sr_img.cpu(), "tmp1.png")

    img_path = "/extra/ucibdl1/shared/projects/conjugate_inverse/kp_results/ground_truth/ffhq/samples/sample_ori_0.png"
    sigma = 3
    window = 61
    kernel = torch.tensor([pdf(t, sigma) for t in range(-(window-1)//2, (window-1)//2)], device="cuda")
    # kernel = torch.Tensor([pdf(-2, sigma), pdf(-1, sigma), pdf(0, sigma), pdf(1, sigma), pdf(2, sigma)]).to(args.device)
    deg = Deblurring(kernel / kernel.sum(), 3, 256, "cuda")
    img = torchvision.io.read_image(img_path).float().to("cuda").unsqueeze(0) / 255.
    db_img = deg.H(img).reshape(3, 256, 256)
    torchvision.utils.save_image(db_img.cpu(), "tmp2.png")

if __name__ == "__main__":
    main()