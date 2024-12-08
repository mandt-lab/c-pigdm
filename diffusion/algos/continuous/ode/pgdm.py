import os

import torch
from torchdiffeq import odeint
from tqdm import tqdm

from utils.degredations import build_degredation_model

from ..base import Sampler


class PGDMSampler(Sampler):
    """
    Implementation of the conditional VP-SDE for inverse problems. Uses the the \lambda-DDIM
    scheme with the PGDM based approximation of the posterior p(x_0|x_t).
    Assumes the noiseless case where \sigma_y = 0
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        if cfg.algo.sigma_y != 0:
            raise ValueError(
                "PGDMSampler sampler only works with zero algo.sigma_y. Use the NoisyConjugatePGDMSampler for a non-zero sigma_y"
            )
        self.is_coeff_computed = False
        self.Phi_soln_list = []
        self.lam = cfg.algo.lam
        self.w = cfg.algo.w
        self.num_eps = cfg.algo.num_eps
        self.st = 0

        # Build degradation model
        self.H = build_degredation_model(cfg)

    def pinv(self, x, shape):
        # H_pinv(x) = H.T inv(HH.T) (x)
        return self.H.H_pinv(x.to(torch.float32)).view(shape)

    def proj(self, x, shape):
        # proj(x) = H.T inv(HH.T) H(x)
        return self.H.H_pinv(self.H.H(x.to(torch.float32))).view(shape)

    def compute_at(self, t):
        int_f_t = -0.5 * self.sde.b_t(t)
        return torch.exp(self.lam * t - int_f_t)

    def At(self, x, t):
        return self.compute_at(t) * x

    def At_inv(self, x, t):
        return torch.reciprocal(self.compute_at(t)) * x

    def compute_coefficients(self, ts):
        def Phi_ode_fn(t, G):
            # \beta_t
            beta_t = self.sde.beta_t(t).type(torch.float64)

            # GG^\top
            gt_sq = beta_t

            # Compute A_t
            a_t = self.compute_at(t)

            # res = 1/2 A_t G_t G_t^\top C_\out(t)
            res = 0.5 * a_t * gt_sq * (1 / (self.sde._std(t) + self.num_eps))
            return res

        Phi_0 = torch.tensor(0, dtype=torch.float64, device=ts.device)
        for t_idx in tqdm(ts):
            time_tensor = torch.tensor(
                [0, self.sde.T - t_idx], dtype=torch.float64, device=ts.device
            )
            Phi_t = odeint(
                Phi_ode_fn,
                Phi_0,
                time_tensor,
                rtol=1e-5,
                atol=1e-5,
                method="scipy_solver",
                options={"solver": "RK45"},
            )
            self.Phi_soln_list.append(Phi_t[-1])

    def initialize(self, x, y0, ts):
        self.st = self.sde.T - ts[0]
        eps = torch.randn_like(x)
        mu_t = self.sde._mean(self.sde.T - ts[0])
        std_t = self.sde._std(self.sde.T - ts[0])
        x0 = self.pinv(y0, x.shape).detach()

        if "inp" in self.cfg.algo.deg:
            return torch.randn_like(x)
        return x0 * mu_t + std_t * eps

    def predictor_update_fn(self, x_bar, y0, label, t, dt, dPhi):
        bs, *_ = x_bar.shape

        # T - t for reverse traversal
        t = self.sde.T - t
        t_ = int(t * (self.sde.N - 1)) * torch.ones(x_bar.shape[0], device=x_bar.device)

        # Compute helper coefficients
        beta_t = self.sde.beta_t(t)
        pinv_y = self.pinv(y0, x_bar.shape)
        xt = self.At_inv(x_bar, t)

        # Compute the jacobian term
        with torch.enable_grad():
            xt = xt.requires_grad_(True)

            # Predict eps and compute the jacobian
            eps_pred = self.model(xt.to(torch.float32), label, t_, scale=1.0)
            x0_pred = self.sde.predict_x_from_eps(xt, eps_pred, t)
            mat = (pinv_y - self.proj(x0_pred, x0_pred.shape)).reshape(bs, -1)
            mat_x = (mat.detach() * x0_pred.reshape(bs, -1)).sum()
            grad_term = torch.autograd.grad(mat_x, xt)[0]

            # Disable further gradients
            x0_pred = x0_pred.detach()
            grad_term = grad_term.detach()
            eps_pred = eps_pred.detach()
            mat = mat.detach()
            xt.requires_grad_(False)

        # Unconditional term
        d_uncond = -self.lam * x_bar * dt + dPhi * eps_pred

        # Conditional term (Assuming w = rt**2)
        d_cond = 0.5 * self.w * beta_t * self.compute_at(t) * grad_term * dt

        # Euler update
        x_bar = x_bar + d_uncond + d_cond
        return x_bar, x0_pred

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        # Get the degraded signal and initialize x
        y0 = kwargs["y_0"]
        xt = self.initialize(x, y0, ts).to(torch.float64)
        assert xt.shape == x.shape

        ts = ts.to(x.device)
        n_steps = ts.size(0) - 1

        # For storing evolution of samples
        xt_s = [xt.cpu()]
        x0_s = []

        # One time coefficient computation (\Phi_t)
        if not self.is_coeff_computed:
            print("Computing ODE coefficients")
            self.compute_coefficients(ts)
            self.is_coeff_computed = True

        # Compute A_T
        t0 = self.sde.T - ts[0]
        x_bar = self.At(xt, t0)

        with torch.no_grad():
            for n in range(n_steps):
                dt = ts[n + 1] - ts[n]
                dPhi = self.Phi_soln_list[n + 1] - self.Phi_soln_list[n]
                x_bar, x0_pred = self.predictor_update_fn(x_bar, y0, y, ts[n], dt, dPhi)
                xt = self.At_inv(x_bar, self.sde.T - ts[n + 1])
                xt_s.append(xt.cpu())
                x0_s.append(x0_pred.cpu())
        return list(reversed(xt_s)), list(reversed(x0_s))
