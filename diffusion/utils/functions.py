import numpy as np
import torch


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def postprocess(x):
    if type(x) == list:
        return [(v + 1) / 2 for v in x]
    else:
        return (x + 1) / 2


def preprocess(x):
    return x * 2 - 1


def get_timesteps(cfg):
    t0 = cfg.exp.t_start
    t1 = cfg.exp.t_end
    stride = cfg.exp.stride
    n_steps = cfg.exp.num_steps

    ts = torch.linspace(t0, t1, n_steps + 1, dtype=torch.float64)

    if stride == "uniform":
        pass
    elif stride == "quadratic":
        rho = 7
        ind = torch.arange(0, n_steps)
        ts = (t0 ** (1/rho) + ind / (n_steps - 1) * (t1 ** (1/rho) - t0 ** (1/rho))) ** rho
    elif stride == "ddpm_uniform":
        # Ensure t0 and t1 are ints
        skip = (t1 - t0) // n_steps
        ts = list(range(t0, t1, skip))
    elif stride == "edm":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return ts


def strfdt(dt):
    days = dt.days
    hours, rem = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds, _ = divmod(dt.microseconds, 1000)

    if days > 0:
        s = f"{days:3d}-"
    else:
        s = "    "
    s += f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return s


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res