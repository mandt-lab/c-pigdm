import torch
from torchdiffeq import odeint
from tqdm import tqdm

from utils.degredations import build_degredation_model

from ..base import Sampler


class ConjugatePGDMSampler(Sampler):
    """
    Implementation of Conjugate PiGDM Sampler for VPSDE with the following design choices:
    1. w_t = w * r_t^2 * \mu_t^2.
    2. \sigma_y = 0
    3. B_t = \lambda I_d
    4. \bar{x}_t = A_t x_t
    5. d\Phi coefficients for efficient integration
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        if cfg.algo.sigma_y != 0:
            raise ValueError(
                "ConjugatePGDM sampler only works with zero algo.sigma_y. Use the NoisyConjugatePGDMSampler for a non-zero sigma_y"
            )
        self.is_coeff_computed = False
        self.lam = cfg.algo.lam
        self.num_eps = cfg.algo.num_eps
        self.w = cfg.algo.w
        self.st = 0
        self.Phiy_soln_list = []
        self.Phis_soln_list = []
        self.Phij_soln_list = []

        # Numerical integration parameters
        self.ode_kws = {
            "rtol": 1e-5,
            "atol": 1e-5,
            "method": "scipy_solver",
            "options": {"solver": "RK45"},
        }

        # Build degradation model
        self.H = build_degredation_model(cfg)
        print(f"Using the C-PiGDM sampler with w: {self.w}, lam: {self.lam}")

    def pinv(self, x, shape):
        # H_pinv(x) = H.T inv(HH.T) (x)
        return self.H.H_pinv(x.to(torch.float32)).view(shape)

    def proj(self, x, shape):
        # proj(x) = H.T inv(HH.T) H(x)
        return self.H.H_pinv(self.H.H(x.to(torch.float32))).view(shape)

    def c1(self, t):
        return self.lam * t + 0.5 * self.sde.b_t(t)

    def c2(self, t):
        return -0.5 * self.w * self.sde.b_t(t)

    def At(self, x, t):
        c1, c2 = self.c1(t), self.c2(t)
        return torch.exp(c1) * (x + (torch.exp(c2) - 1) * self.proj(x, x.shape))

    def At_inv(self, x, t):
        c1, c2 = self.c1(t), self.c2(t)
        return torch.exp(-c1) * (x + (torch.exp(-c2) - 1) * self.proj(x, x.shape))

    def compute_coefficients(self, ts):
        def Phiy_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            mu_t = self.sde._mean(t)
            c1 = self.c1(t)
            c2 = self.c2(t)
            return -0.5 * self.w * beta_t * torch.exp(c1 + c2) * mu_t

        def Phis1_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            c1 = self.c1(t)
            return 0.5 * beta_t * torch.exp(c1) / (std_t + self.num_eps)

        def Phis2_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            c1, c2 = self.c1(t), self.c2(t)
            res1 = (
                0.5
                * beta_t
                * torch.exp(c1)
                * (torch.exp(c2) - 1)
                / (std_t + self.num_eps)
            )
            res2 = -0.5 * self.w * beta_t * std_t * torch.exp(c1 + c2)
            return res1 + res2

        def Phij1_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            mu_t = self.sde._mean(t)
            c1 = self.c1(t)
            return 0.5 * self.w * beta_t * torch.exp(c1) * std_t * mu_t

        def Phij2_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            mu_t = self.sde._mean(t)
            c1 = self.c1(t)
            c2 = self.c2(t)
            return (
                0.5
                * self.w
                * beta_t
                * std_t
                * torch.exp(c1)
                * (torch.exp(c2) - 1)
                * mu_t
            )

        zero_t = torch.tensor(0, dtype=torch.float64, device=ts.device)
        Phiy_prev = Phis1_prev = Phis2_prev = Phij1_prev = Phij2_prev = zero_t
        prev_t = 0
        for t_idx in tqdm(reversed(ts)):
            time_tensor = torch.tensor(
                [prev_t, self.sde.T - t_idx], dtype=torch.float64, device=ts.device
            )
            # Compute coefficients
            Phiy = odeint(Phiy_ode_fn, Phiy_prev, time_tensor, **self.ode_kws)
            Phis1 = odeint(Phis1_ode_fn, Phis1_prev, time_tensor, **self.ode_kws)
            Phis2 = odeint(Phis2_ode_fn, Phis2_prev, time_tensor, **self.ode_kws)
            Phij1 = odeint(Phij1_ode_fn, Phij1_prev, time_tensor, **self.ode_kws)
            Phij2 = odeint(Phij2_ode_fn, Phij2_prev, time_tensor, **self.ode_kws)

            self.Phiy_soln_list.insert(0, Phiy[-1])
            self.Phis_soln_list.insert(0, torch.tensor([Phis1[-1], Phis2[-1]]))
            self.Phij_soln_list.insert(0, torch.tensor([Phij1[-1], Phij2[-1]]))

            # Update initial values
            prev_t = self.sde.T - t_idx
            Phiy_prev = Phiy[-1]
            Phis1_prev, Phis2_prev = Phis1[-1], Phis2[-1]
            Phij1_prev, Phij2_prev = Phij1[-1], Phij2[-1]

    def initialize(self, x, y0, ts):
        self.st = self.sde.T - ts[0]
        eps = torch.randn_like(x)
        mu_t = self.sde._mean(self.sde.T - ts[0])
        std_t = self.sde._std(self.sde.T - ts[0])
        x0 = self.pinv(y0, x.shape).detach()

        # if 'inp' in self.cfg.algo.deg:
        #     return torch.randn_like(x)
        return x0 * mu_t + std_t * eps

    def predictor_update_fn(self, x_bar, xt, y, label, t, dt, dPhiy, dPhis, dPhij):
        bs, *_ = x_bar.shape

        # T - t for reverse traversal
        t = self.sde.T - t
        t_ = int(t * (self.sde.N - 1)) * torch.ones(x_bar.shape[0], device=x_bar.device)

        # Compute helper coefficients
        pinv_y = self.pinv(y, x_bar.shape)

        with torch.enable_grad():
            # Enable gradients wrt xt
            xt = xt.requires_grad_(True)

            # Predict eps and compute the jacobian
            eps_pred = self.model(xt.to(torch.float32), label, t_, scale=1.0)
            x0_pred = self.sde.predict_x_from_eps(xt, eps_pred, t)
            mat = (pinv_y - self.proj(x0_pred, x0_pred.shape)).reshape(bs, -1)
            mat_x = (mat.detach() * eps_pred.reshape(bs, -1)).sum()
            grad_term = torch.autograd.grad(mat_x, xt)[0]

            # Disable further gradients
            x0_pred = x0_pred.detach()
            grad_term = grad_term.detach()
            eps_pred = eps_pred.detach()
            mat = mat.detach()
            xt.requires_grad_(False)

        # x_bar term
        d_xbar = -self.lam * x_bar * dt

        # score terms
        dPhis0, dPhis1 = dPhis[0], dPhis[1]
        d_score = dPhis0 * eps_pred + dPhis1 * self.proj(eps_pred, eps_pred.shape)

        # y term
        d_y = dPhiy * pinv_y

        # Jacobian term
        dPhij0, dPhij1 = dPhij[0], dPhij[1]
        d_jacobian = dPhij0 * grad_term + dPhij1 * self.proj(grad_term, grad_term.shape)

        # Final update
        dx = d_xbar + d_y + d_score + d_jacobian
        x_bar = x_bar + dx
        return x_bar, x0_pred

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        # Get the degraded signal and initialize x
        y0 = kwargs["y_0"]
        x = self.initialize(x, y0, ts).to(torch.float64)
        ts = ts.to(x.device)
        n_steps = ts.size(0) - 1

        # For storing evolution of samples
        xt_s = [x.cpu()]
        x_bars = []
        x0_s = []

        # One time coefficient computation (\Phi_t)
        if not self.is_coeff_computed:
            print("Computing ODE coefficients")
            self.compute_coefficients(ts)
            self.is_coeff_computed = True

        # Compute A_T
        t0 = self.sde.T - ts[0]
        x_bar = self.At(x, t0)
        x_bars.append(x_bar.cpu())

        with torch.no_grad():
            for n in range(n_steps):
                dt = ts[n + 1] - ts[n]
                dPhiy = self.Phiy_soln_list[n + 1] - self.Phiy_soln_list[n]
                dPhis = self.Phis_soln_list[n + 1] - self.Phis_soln_list[n]
                dPhij = self.Phij_soln_list[n + 1] - self.Phij_soln_list[n]
                x_bar, x0_pred = self.predictor_update_fn(
                    x_bar, x, y0, y, ts[n], dt, dPhiy, dPhis, dPhij
                )

                # Map back to x and save trajectory
                x = self.At_inv(x_bar, self.sde.T - ts[n + 1])
                xt_s.append(x.cpu())
                x0_s.append(x0_pred.cpu())
        return list(reversed(xt_s)), list(reversed(x0_s))


class NoisyConjugatePGDMSampler(Sampler):
    """
    Implementation of Conjugate PGDM Sampler with the following design choices:
    1. w_t = w * r_t^2 * \mu_t^2.
    2. \sigma_y can be non-zero
    3. B_t = \lambda I_d
    4. \bar{x}_t = A_t x_t
    5. d\Phi coefficients for efficient integration
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.is_coeff_computed = False
        self.lam = cfg.algo.lam
        self.num_eps = cfg.algo.num_eps
        self.w = cfg.algo.w
        self.sigma_y = cfg.algo.sigma_y
        self.y0_shape = None
        self.st = 0
        self.Phiy_soln_list = []
        self.Phis_soln_list = []
        self.Phij_soln_list = []
        self.Phic_soln_list = []

        # Numerical integration parameters
        self.ode_kws = {
            "rtol": 1e-5,
            "atol": 1e-5,
            "method": "scipy_solver",
            "options": {"solver": "RK45"},
        }

        # Build degradation model
        self.H = build_degredation_model(cfg)
        print(
            f"Using the Noisy C-PiGDM sampler with w: {self.w}, lam: {self.lam}, sigma_y: {self.sigma_y}"
        )

    def pinv(self, x, shape):
        # H_pinv(x) = H.T inv(HH.T) (x)
        return self.H.H_pinv(x.to(torch.float32)).view(shape)

    def pinvt(self, x, shape):
        # H_pinv(x) = H.T inv(HH.T) (x)
        return self.H.H_pinvt(x.to(torch.float32)).view(shape)

    def proj(self, x, shape):
        # proj(x) = H.T inv(HH.T) H(x)
        return self.H.H_pinv(self.H.H(x.to(torch.float32))).view(shape)

    def c1(self, t):
        return self.lam * t + 0.5 * self.sde.b_t(t)

    def c2(self, t):
        return -0.5 * self.w * self.sde.b_t(t)

    def c3(self, t):
        c1, c2 = self.c1(t), self.c2(t)
        b_t = self.sde.b_t(t)
        temp = torch.log(torch.exp(b_t) - 1 + self.num_eps)
        return 0.5 * self.w * (torch.exp(c1 + c2) - 1) * (self.sigma_y**2) * temp

    def At(self, x, t):
        c1, c2 = self.c1(t), self.c2(t)
        return torch.exp(c1) * (x + (torch.exp(c2) - 1) * self.proj(x, x.shape))

    def At_noisy(self, x, t):
        At0 = self.At(x, t)
        c3 = self.c3(t)
        return At0 + c3 * self.pinv(self.pinvt(x, self.y0_shape), x.shape)

    def At_inv(self, x, t):
        c1, c2 = self.c1(t), self.c2(t)
        return torch.exp(-c1) * (x + (torch.exp(-c2) - 1) * self.proj(x, x.shape))

    def At_inv_noisy(self, x, t):
        At0_inv = self.At_inv(x, t)
        c3 = self.c3(t)
        temp = c3 * self.pinv(self.pinvt(At0_inv, self.y0_shape), x.shape)
        return At0_inv - self.At_inv(temp, t)

    def compute_coefficients(self, ts):
        def Phiy_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            mu_t = self.sde._mean(t)
            c1, c2 = self.c1(t), self.c2(t)
            return -0.5 * self.w * beta_t * torch.exp(c1 + c2) * mu_t

        def Phiyc_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            mu_t = self.sde._mean(t)
            c3 = self.c3(t)
            return -0.5 * beta_t * self.w * mu_t * c3

        def Phis1_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            c1 = self.c1(t)
            return 0.5 * beta_t * torch.exp(c1) / (std_t + self.num_eps)

        def Phis2_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            c1, c2 = self.c1(t), self.c2(t)
            res1 = (
                0.5
                * beta_t
                * torch.exp(c1)
                * (torch.exp(c2) - 1)
                / (std_t + self.num_eps)
            )
            res2 = -0.5 * self.w * beta_t * std_t * torch.exp(c1 + c2)
            return res1 + res2

        def Phisc_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            c3 = self.c3(t)
            return (
                0.5
                * beta_t
                * c3
                * (torch.reciprocal(std_t + self.num_eps) - self.w * std_t)
            )

        def Phij1_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            mu_t = self.sde._mean(t)
            c1 = self.c1(t)
            return 0.5 * self.w * beta_t * torch.exp(c1) * std_t * mu_t

        def Phij2_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            mu_t = self.sde._mean(t)
            c1, c2 = self.c1(t), self.c2(t)
            return (
                0.5
                * self.w
                * beta_t
                * std_t
                * torch.exp(c1)
                * (torch.exp(c2) - 1)
                * mu_t
            )

        def Phijc_ode_fn(t, G):
            beta_t = self.sde.beta_t(t)
            std_t = self.sde._std(t)
            mu_t = self.sde._mean(t)
            c3 = self.c3(t)
            return 0.5 * self.w * std_t * mu_t * beta_t * c3

        zero_t = torch.tensor(0, dtype=torch.float64, device=ts.device)
        Phiy_prev = Phis1_prev = Phis2_prev = Phij1_prev = Phij2_prev = zero_t
        Phiyc_prev = Phisc_prev = Phijc_prev = zero_t
        prev_t = 0
        for t_idx in tqdm(reversed(ts)):
            time_tensor = torch.tensor(
                [prev_t, self.sde.T - t_idx], dtype=torch.float64, device=ts.device
            )
            # Compute coefficients
            Phiy = odeint(Phiy_ode_fn, Phiy_prev, time_tensor, **self.ode_kws)
            Phis1 = odeint(Phis1_ode_fn, Phis1_prev, time_tensor, **self.ode_kws)
            Phis2 = odeint(Phis2_ode_fn, Phis2_prev, time_tensor, **self.ode_kws)
            Phij1 = odeint(Phij1_ode_fn, Phij1_prev, time_tensor, **self.ode_kws)
            Phij2 = odeint(Phij2_ode_fn, Phij2_prev, time_tensor, **self.ode_kws)
            Phiyc = odeint(Phiyc_ode_fn, Phiyc_prev, time_tensor, **self.ode_kws)
            Phisc = odeint(Phisc_ode_fn, Phisc_prev, time_tensor, **self.ode_kws)
            Phijc = odeint(Phijc_ode_fn, Phijc_prev, time_tensor, **self.ode_kws)

            self.Phiy_soln_list.insert(0, Phiy[-1])
            self.Phis_soln_list.insert(0, torch.tensor([Phis1[-1], Phis2[-1]]))
            self.Phij_soln_list.insert(0, torch.tensor([Phij1[-1], Phij2[-1]]))
            self.Phic_soln_list.insert(
                0, torch.tensor([Phiyc[-1], Phisc[-1], Phijc[-1]])
            )

            # Update initial values
            prev_t = self.sde.T - t_idx
            Phiy_prev = Phiy[-1]
            Phis1_prev, Phis2_prev = Phis1[-1], Phis2[-1]
            Phij1_prev, Phij2_prev = Phij1[-1], Phij2[-1]
            Phiyc_prev, Phisc_prev, Phijc_prev = Phiyc[-1], Phisc[-1], Phijc[-1]

    def initialize(self, x, y0, ts):
        if "inp" in self.cfg.algo.deg:
            return torch.randn_like(x)
        self.st = self.sde.T - ts[0]
        eps = torch.randn_like(x)
        mu_t = self.sde._mean(self.sde.T - ts[0])
        std_t = self.sde._std(self.sde.T - ts[0])
        x0 = self.pinv(y0, x.shape).detach()
        return x0 * mu_t + std_t * eps

    def predictor_update_fn(self, x_bar, y, label, t, dt, dPhiy, dPhis, dPhij, dPhic):
        bs, *_ = x_bar.shape

        # T - t for reverse traversal
        t = self.sde.T - t
        t_ = int(t * (self.sde.N - 1)) * torch.ones(x_bar.shape[0], device=x_bar.device)

        # Compute helper coefficients
        pinv_y = self.pinv(y, x_bar.shape)
        xt = self.At_inv_noisy(x_bar, t)

        with torch.enable_grad():
            # Enable gradients wrt xt
            xt = xt.requires_grad_(True)

            # Predict eps and compute the jacobian
            eps_pred = self.model(xt.to(torch.float32), label, t_, scale=1.0)
            x0_pred = self.sde.predict_x_from_eps(xt, eps_pred, t)
            mat = (pinv_y - self.proj(x0_pred, x0_pred.shape)).reshape(bs, -1)
            mat_x = (mat.detach() * eps_pred.reshape(bs, -1)).sum()
            grad_term = torch.autograd.grad(mat_x, xt)[0]

            # Disable further gradients
            x0_pred = x0_pred.detach()
            grad_term = grad_term.detach()
            eps_pred = eps_pred.detach()
            mat = mat.detach()
            xt.requires_grad_(False)

        shape = eps_pred.shape
        dPhiyc, dPhisc, dPhijc = dPhic[0], dPhic[1], dPhic[2]

        # x_bar term
        d_xbar = -self.lam * x_bar * dt

        # score terms
        dPhis0, dPhis1 = dPhis[0], dPhis[1]
        d_score = dPhis0 * eps_pred + dPhis1 * self.proj(eps_pred, shape)
        d_score = d_score + dPhisc * self.pinv(
            self.pinvt(eps_pred, self.y0_shape), shape
        )

        # y term
        d_y = dPhiy * pinv_y
        d_y = d_y + dPhiyc * self.pinv(self.pinvt(pinv_y, self.y0_shape), shape)

        # Jacobian term
        dPhij0, dPhij1 = dPhij[0], dPhij[1]
        d_jacobian = dPhij0 * grad_term + dPhij1 * self.proj(grad_term, grad_term.shape)
        d_jacobian = d_jacobian + dPhijc * self.pinv(
            self.pinvt(grad_term, self.y0_shape), shape
        )

        # Final update
        x_bar = x_bar + (d_xbar + d_y + d_score + d_jacobian)
        return x_bar, x0_pred

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        # Get the degraded signal and initialize x
        y0 = kwargs["y_0"]
        self.y0_shape = y0.shape
        x = self.initialize(x, y0, ts).to(torch.float64)
        ts = ts.to(x.device)
        n_steps = ts.size(0) - 1

        # For storing evolution of samples
        xt_s = [x.cpu()]
        x0_s = []

        # One time coefficient computation (\Phi_t)
        if not self.is_coeff_computed:
            print("Computing ODE coefficients")
            self.compute_coefficients(ts)
            self.is_coeff_computed = True

        # Compute A_T
        t0 = self.sde.T - ts[0]
        x_bar = self.At_noisy(x, t0)

        with torch.no_grad():
            for n in range(n_steps):
                dt = ts[n + 1] - ts[n]
                dPhiy = self.Phiy_soln_list[n + 1] - self.Phiy_soln_list[n]
                dPhis = self.Phis_soln_list[n + 1] - self.Phis_soln_list[n]
                dPhij = self.Phij_soln_list[n + 1] - self.Phij_soln_list[n]
                dPhic = self.Phic_soln_list[n + 1] - self.Phic_soln_list[n]
                x_bar, x0_pred = self.predictor_update_fn(
                    x_bar, y0, y, ts[n], dt, dPhiy, dPhis, dPhij, dPhic
                )

                # Map back to x and save trajectory
                x = self.At_inv_noisy(x_bar, self.sde.T - ts[n + 1])
                xt_s.append(x.cpu())
                x0_s.append(x0_pred.cpu())
        return list(reversed(xt_s)), list(reversed(x0_s))
