import torch
from torchdiffeq import odeint
from tqdm import tqdm

from ..base import Sampler


class VPConjugateSampler(Sampler):
    """
    Implementation of the unconditional conjugate integrator with VPSDE.
    This is equivalent to the \lambda-DDIM proposed in https://arxiv.org/abs/2310.07894
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.is_coeff_computed = False
        self.Phi_soln_list = []
        self.lam = cfg.algo.lam
        self.num_eps = cfg.algo.num_eps

    def convert_xbar_to_x(self, x_bar, a_t):
        return x_bar / a_t

    def convert_x_to_xbar(self, x, a_t):
        return x * a_t

    def compute_at(self, t):
        int_f_t = -0.5 * self.sde.b_t(t)
        return torch.exp(self.lam * t - int_f_t)

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

    def predictor_update_fn(self, x_bar, y, t, dt, dPhi):
        t = self.sde.T - t
        ones = torch.ones(x_bar.shape[0], device=x_bar.device, dtype=torch.float64)

        # Predict eps (continuous time is discretized here since we work with ADM diffusion models)
        a_t = self.compute_at(t)
        x = self.convert_xbar_to_x(x_bar, a_t)
        eps_pred, x0_pred = self.model(
            x, y, int(t * (self.sde.N - 1)) * ones, scale=1.0
        )

        # Update
        d_x = -self.lam * x_bar * dt + dPhi * eps_pred
        x_bar = x_bar + d_x
        return x_bar, x0_pred

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        # NOTE: This is an unconditional sampler so the input degraded signal x is not used anyways.
        x = torch.randn_like(x)
        n_steps = ts.size(0) - 1

        # For storing evolution of samples
        xt_s = [x.cpu()]
        x0_s = []

        # One time coefficient computation (\Phi_t)
        if not self.is_coeff_computed:
            print("Computing ODE coefficients")
            self.compute_coefficients(ts)
            self.is_coeff_computed = True

        # Compute A_T and project
        t_0 = self.sde.T - ts[0]
        a_T = self.compute_at(t_0)
        x_bar = self.convert_x_to_xbar(x, a_T)

        with torch.no_grad():
            for n in range(n_steps):
                dPhi = self.Phi_soln_list[n + 1] - self.Phi_soln_list[n]
                dt = ts[n + 1] - ts[n]
                x_bar, x0_pred = self.predictor_update_fn(x_bar, y, ts[n], dt, dPhi)

                # Map back to x and save trajectory
                curr_a = self.compute_at(self.sde.T - ts[n + 1])
                x = self.convert_xbar_to_x(x_bar, curr_a)
                xt_s.append(x.cpu())
                x0_s.append(x0_pred.cpu())

            if kwargs.get("denoise", False):
                dPhi = self.Phi_soln_list[-1]
                x_bar, x0_pred = self.predictor_update_fn(x_bar, y, ts[-1], dt, dPhi)
                t_last = self.sde.T - ts[-1]
                curr_a = self.compute_at(t_last)
                x = self.convert_xbar_to_x(x_bar, curr_a)
                xt_s.append(x.cpu())
                x0_s.append(x0_pred.cpu())
        return list(reversed(xt_s)), list(reversed(x0_s))
