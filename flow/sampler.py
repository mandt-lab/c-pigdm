import torch
from torchdiffeq import odeint
from tqdm import tqdm

class PGDM:
    def __init__(self, v_theta, deg, skip_t=0.1, w=1):
        self.v_theta = v_theta
        self.w = w
        self.deg = deg
        self.device = self.v_theta.device
        self.st = skip_t * torch.ones(1, device=self.device)

    def proj(self, x):
        return self.deg.H_pinv(self.deg.H(x))

    def H(self, x):
        return self.deg.H(x)

    def H_pinv(self, x):
        return self.deg.H_pinv(x)

    @torch.no_grad()
    def sample_pgdm(self, y, img_dim, n_steps=40):
        bs = y.shape[0]
        y1 = self.H_pinv(y)
        xt = self.st * y1 + (1 - self.st) * torch.randn_like(y1)
        dt = (1 - self.st) / n_steps
        for i in tqdm(range(n_steps)):
            t = self.st + i * dt
            # pgdm direction
            with torch.enable_grad():
                xt.requires_grad_(True)
                xt_tmp = xt.reshape(bs, *img_dim)
                vt = self.v_theta(xt_tmp, t * torch.ones(bs, device=self.device) * 999)
                vt = vt.reshape(bs, -1)
                xt_tmp = xt_tmp.reshape(bs, -1)
                pred_x1 = xt_tmp + (1 - t) * vt
                mat = (y1 - self.H_pinv(self.H(pred_x1))).reshape(bs, -1)
                mat_x = (mat.detach() * pred_x1.reshape(bs, -1)).sum()
                grad_term = torch.autograd.grad(mat_x, xt)[0]
                xt.requires_grad_(False)
            vt_new = vt + ((1 - t) ** 2 + t**2) / (t * (1 - t)) * self.w * grad_term
            # vt_new = vt + (1-t) / t * self.w * grad_term
            xt = xt + dt * vt_new
        return xt.reshape(bs, *img_dim)


class Conjugate:
    def __init__(self, v_theta, deg, integral_start=1e-6, skip_t=0.1, w=1, B=1):
        self.v_theta = v_theta
        self.w = w
        self.deg = deg
        self.device = self.v_theta.device
        self.st = skip_t * torch.ones(1, device=self.device)
        self.B = B
        self.init = torch.zeros(1, device=self.device, dtype=torch.float64)
        self.integral_start = integral_start * torch.ones_like(self.st)

    def get_a(self, t):
        a = self.B * (t - self.integral_start)
        return a
    
    def get_b(self, t):
        b = self.w * (torch.log(t) - torch.log(1 - t) - torch.log(self.integral_start) + torch.log(1 - self.integral_start) - 2 * t + 2 * self.integral_start)
        return b

    def At(self, x, t, x_projected=False):
        # return x
        a = self.get_a(t)
        b = self.get_b(t)
        if x_projected:
            return torch.exp(a) * (x + (torch.exp(b) - 1) * x)
        return torch.exp(a) * (x + (torch.exp(b) - 1) * self.proj(x))
    
    def At_inv(self, x, t):
        # return x
        a = self.get_a(t)
        b = self.get_b(t)
        return torch.exp(-a) * (x + (torch.exp(-b) - 1) * self.proj(x))

    # def coeff_intAt(self, t):
    #     if torch.abs(t - self.integral_start) < 1e-6:
    #         return 0, 0
    #     def fn1(t):
    #         a = self.get_a(t)
    #         if self.B == 0:
    #             return 0
    #         return torch.exp(a) / self.B
    #     def ode_fn2(t, y):
    #         a = self.get_a(t)
    #         b = self.get_b(t)
    #         return torch.exp(a) * (torch.exp(b) - 1)
    #     p1 = fn1(t) - fn1(self.integral_start)
    #     p2 = odeint(
    #         ode_fn2,
    #         self.init,
    #         torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
    #         rtol=1e-5,
    #         atol=1e-5,
    #         method="scipy_solver",
    #         options={"solver": "RK45"},
    #     )[-1]
    #     return p1, p2
    
    def coeff_intAt(self, t):
        if torch.abs(t - self.integral_start) < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2
    
    def coeff_phi1(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0, 0, 0
            
        intAt_p1, intAt_p2 = self.coeff_intAt(t)
        def ode_fn3(t, y):
            a = self.get_a(t)
            return self.w * ((1 - t) ** 2 + t**2) * torch.reciprocal(t) * torch.exp(a)
        def ode_fn4(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * ((1 - t) ** 2 + t**2) * torch.reciprocal(t) * torch.exp(a) * (torch.exp(b) - 1)
        p3 = odeint(
            ode_fn3,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p4 = odeint(
            ode_fn4,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]

        return intAt_p1, intAt_p2, p3, p4
    
    def coeff_phi2(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return self.w * ((1 - t) ** 2 + t**2) * torch.reciprocal(t * (1-t)) * torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * ((1 - t) ** 2 + t**2) * torch.reciprocal(t * (1-t)) * torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2

    def proj(self, x):
        return self.deg.H_pinv(self.deg.H(x))

    def H(self, x):
        return self.deg.H(x)

    def H_pinv(self, x):
        return self.deg.H_pinv(x)
    
    def get_coeff(self, n_steps):
        dt = (1 - self.st) / n_steps
        coeffs = []
        for i in torch.arange(n_steps + 1):
            t = (self.st + i * dt).clamp(max = 1)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4 = self.coeff_phi1(t)
            ph2_p1, ph2_p2 = self.coeff_phi2(t)
            coeffs.append([ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2])
        return coeffs

    @torch.no_grad()
    def sample_conj(self, y, img_dim, n_steps=40):
        bs = y.shape[0]
        y1 = self.H_pinv(y)
        xt = self.st * y1 + (1 - self.st) * torch.randn_like(y1)
        xt_bar = self.At(xt, self.st)
        dt = (1 - self.st) / n_steps
        coeffs = self.get_coeff(n_steps)
        for i in tqdm(range(n_steps)):
            t = self.st + i * dt
            xt = self.At_inv(xt_bar, t)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2 = coeffs[i]
            ph1_intAt_p1_plus, ph1_intAt_p2_plus, ph1_p3_plus, ph1_p4_plus, ph2_p1_plus, ph2_p2_plus = coeffs[i+1]
            t_tensor = t * torch.ones(bs, device=self.device)

            # jacobian term
            with torch.enable_grad():
                xt.requires_grad_(True)
                tmp_xt = xt.reshape(bs, *img_dim)
                vt = self.v_theta(tmp_xt, t_tensor * 999)
                vt = vt.reshape(bs, -1)
                tmp_xt = tmp_xt.reshape(bs, -1)
                pred_x1 = tmp_xt + (1 - t) * vt
                mat = (y1 - self.proj(pred_x1)).reshape(bs, -1)
                mat_x = (mat.detach() * vt).sum()
                # print(mat_x.requires_grad, vt.requires_grad, xt.requires_grad)
                grad_term = torch.autograd.grad(mat_x, xt)[0]
                # print(grad_term)
                xt.requires_grad_(False)
                
            dgrad_term = dt * self.At(grad_term, t, True) * self.w * ((1 - t) ** 2 + t**2) * torch.reciprocal(t)

            # v_theta term
            # proj_vt = self.proj(vt)
            # dv = (ph1_intAt_p1_plus - ph1_intAt_p1) * vt + (ph1_intAt_p2_plus - ph1_intAt_p2) * proj_vt - (ph1_p3_plus - ph1_p3) * proj_vt - (ph1_p4_plus - ph1_p4) * self.proj(proj_vt)

            # v_theta term non-integral
            dv = dt * (self.At(vt, t) - self.w * (t**2 + (1-t)**2) * torch.reciprocal(t) * self.At(self.proj(vt), t, True))

            # y term
            # dy = (ph2_p1_plus - ph2_p1) * y1 + (ph2_p2_plus - ph2_p2) * self.proj(y1)

            # y term non-integral
            dy = dt * (self.w * (t**2 + (1-t)**2) * torch.reciprocal(t * (1-t)) * self.At(y1, t, True))

            # head term
            dhead = self.B * xt_bar * dt

            # dhead = - self.w * (t**2 + (1-t)**2) * torch.reciprocal(t * (1-t)) * self.At(self.proj(xt_bar), t) * dt

            dxt_bar = dhead + dv + dy + dgrad_term
            xt_bar = (xt_bar + dxt_bar).float()
        xt = self.At_inv(xt_bar, torch.ones_like(self.st))
        return xt.reshape(bs, *img_dim)

class ConjugateV2(Conjugate):
    def get_b(self, t):
        b = self.w * (t**2/2 - t**3/3 - self.integral_start**2/2 + self.integral_start**3/3)
        return b
    
    def coeff_phi1(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0, 0, 0
            
        intAt_p1, intAt_p2 = self.coeff_intAt(t)
        def ode_fn3(t, y):
            a = self.get_a(t)
            return self.w * (1-t)**2 * t * torch.exp(a)
        def ode_fn4(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t)**2 * t * torch.exp(a) * (torch.exp(b) - 1)
        p3 = odeint(
            ode_fn3,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p4 = odeint(
            ode_fn4,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]

        return intAt_p1, intAt_p2, p3, p4
    
    def coeff_phi2(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return self.w * (1-t) * t * torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t) * t * torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2
    
    def coeff_phi3(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return self.w * (1-t)**2 * t * torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t)**2 * t * torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2
    
    def get_coeff(self, n_steps):
        dt = (1 - self.st) / n_steps
        coeffs = []
        for i in torch.arange(n_steps + 1):
            t = (self.st + i * dt).clamp(max = 1)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4 = self.coeff_phi1(t)
            ph2_p1, ph2_p2 = self.coeff_phi2(t)
            ph3_p1, ph3_p2 = self.coeff_phi3(t)
            coeffs.append([ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2, ph3_p1, ph3_p2])
        return coeffs

    
    @torch.no_grad()
    def sample_conj(self, y, img_dim, n_steps=40):
        bs = y.shape[0]
        y1 = self.H_pinv(y)
        xt = self.st * y1 + (1 - self.st) * torch.randn_like(y1)
        xt_bar = self.At(xt, self.st)
        dt = (1 - self.st) / n_steps
        coeffs = self.get_coeff(n_steps)
        for i in tqdm(range(n_steps)):
            t = self.st + i * dt
            xt = self.At_inv(xt_bar, t)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2, ph3_p1, ph3_p2 = coeffs[i]
            ph1_intAt_p1_plus, ph1_intAt_p2_plus, ph1_p3_plus, ph1_p4_plus, ph2_p1_plus, ph2_p2_plus, ph3_p1_plus, ph3_p2_plus = coeffs[i+1]
            t_tensor = t * torch.ones(bs, device=self.device)

            # jacobian term
            with torch.enable_grad():
                xt.requires_grad_(True)
                tmp_xt = xt.reshape(bs, *img_dim)
                vt = self.v_theta(tmp_xt, t_tensor * 999)
                vt = vt.reshape(bs, -1)
                tmp_xt = tmp_xt.reshape(bs, -1)
                pred_x1 = tmp_xt + (1 - t) * vt
                mat = (y1 - self.proj(pred_x1)).reshape(bs, -1)
                mat_x = (mat.detach() * vt).sum()
                # print(mat_x.requires_grad, vt.requires_grad, xt.requires_grad)
                grad_term = torch.autograd.grad(mat_x, xt)[0]
                # print(grad_term)
                xt.requires_grad_(False)

            dgrad_term = dt * (ph3_p1_plus - ph3_p1) * grad_term + dt * (ph3_p2_plus - ph3_p2) * grad_term

            # gradient term non-integral
            # dgrad_term = dt * self.At(grad_term, t, True) * self.w * (1-t)**2 * t

            # v_theta term
            proj_vt = self.proj(vt)
            dv = (ph1_intAt_p1_plus - ph1_intAt_p1) * vt + (ph1_intAt_p2_plus - ph1_intAt_p2) * proj_vt - (ph1_p3_plus - ph1_p3) * proj_vt - (ph1_p4_plus - ph1_p4) * proj_vt

            # v_theta term non-integral
            # dv = dt * (self.At(vt, t) - self.w * (1-t)**2 * t * self.At(self.proj(vt), t, True))

            # y term
            dy = (ph2_p1_plus - ph2_p1) * y1 + (ph2_p2_plus - ph2_p2) * y1

            # y term non-integral
            # dy = dt * self.w * (1-t)**2 * t * self.At(y1, t, True)

            # head term
            dhead = self.B * xt_bar * dt

            dxt_bar = dhead + dv + dy + dgrad_term
            xt_bar = (xt_bar + dxt_bar).float()
        xt = self.At_inv(xt_bar, torch.ones_like(self.st))
        return xt.reshape(bs, *img_dim)
    
class ConjugateV3(Conjugate):
    def get_b(self, t):
        b = self.w * (t - t**2/2 - self.integral_start + self.integral_start**2/2)
        return b
    
    def coeff_phi1(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0, 0, 0
            
        intAt_p1, intAt_p2 = self.coeff_intAt(t)
        def ode_fn3(t, y):
            a = self.get_a(t)
            return self.w * (1-t)**2 * torch.exp(a)
        def ode_fn4(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t)**2 * torch.exp(a) * (torch.exp(b) - 1)
        p3 = odeint(
            ode_fn3,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p4 = odeint(
            ode_fn4,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]

        return intAt_p1, intAt_p2, p3, p4
    
    def coeff_phi2(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return self.w * (1-t) * torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t) * torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2
    
    def coeff_phi3(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return self.w * (1-t)**2 * torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t)**2 * torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2
    
    def get_coeff(self, n_steps):
        dt = (1 - self.st) / n_steps
        coeffs = []
        for i in torch.arange(n_steps + 1):
            t = (self.st + i * dt).clamp(max = 1)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4 = self.coeff_phi1(t)
            ph2_p1, ph2_p2 = self.coeff_phi2(t)
            ph3_p1, ph3_p2 = self.coeff_phi3(t)
            coeffs.append([ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2, ph3_p1, ph3_p2])
        return coeffs

    
    @torch.no_grad()
    def sample_conj(self, y, img_dim, n_steps=40):
        bs = y.shape[0]
        y1 = self.H_pinv(y)
        xt = self.st * y1 + (1 - self.st) * torch.randn_like(y1)
        xt_bar = self.At(xt, self.st)
        dt = (1 - self.st) / n_steps
        coeffs = self.get_coeff(n_steps)
        for i in tqdm(range(n_steps)):
            t = self.st + i * dt
            xt = self.At_inv(xt_bar, t)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2, ph3_p1, ph3_p2 = coeffs[i]
            ph1_intAt_p1_plus, ph1_intAt_p2_plus, ph1_p3_plus, ph1_p4_plus, ph2_p1_plus, ph2_p2_plus, ph3_p1_plus, ph3_p2_plus = coeffs[i+1]
            t_tensor = t * torch.ones(bs, device=self.device)

            # jacobian term
            with torch.enable_grad():
                xt.requires_grad_(True)
                tmp_xt = xt.reshape(bs, *img_dim)
                vt = self.v_theta(tmp_xt, t_tensor * 999)
                vt = vt.reshape(bs, -1)
                tmp_xt = tmp_xt.reshape(bs, -1)
                pred_x1 = tmp_xt + (1 - t) * vt
                mat = (y1 - self.proj(pred_x1)).reshape(bs, -1)
                mat_x = (mat.detach() * vt).sum()
                # print(mat_x.requires_grad, vt.requires_grad, xt.requires_grad)
                grad_term = torch.autograd.grad(mat_x, xt)[0]
                # print(grad_term)
                xt.requires_grad_(False)

            dgrad_term = dt * (ph3_p1_plus - ph3_p1) * grad_term + dt * (ph3_p2_plus - ph3_p2) * grad_term

            # gradient term non-integral
            # dgrad_term = dt * self.At(grad_term, t, True) * self.w * (1-t)**2 * t

            # v_theta term
            proj_vt = self.proj(vt)
            dv = (ph1_intAt_p1_plus - ph1_intAt_p1) * vt + (ph1_intAt_p2_plus - ph1_intAt_p2) * proj_vt - (ph1_p3_plus - ph1_p3) * proj_vt - (ph1_p4_plus - ph1_p4) * proj_vt

            # v_theta term non-integral
            # dv = dt * (self.At(vt, t) - self.w * (1-t)**2 * t * self.At(self.proj(vt), t, True))

            # y term
            dy = (ph2_p1_plus - ph2_p1) * y1 + (ph2_p2_plus - ph2_p2) * y1

            # y term non-integral
            # dy = dt * self.w * (1-t)**2 * t * self.At(y1, t, True)

            # head term
            dhead = self.B * xt_bar * dt

            dxt_bar = dhead + dv + dy + dgrad_term
            xt_bar = (xt_bar + dxt_bar).float()
        xt = self.At_inv(xt_bar, torch.ones_like(self.st))
        return xt.reshape(bs, *img_dim)

class ConjugateV4(Conjugate):
    def get_b(self, t):
        b = self.w * (torch.log(t) - t - torch.log(self.integral_start) + self.integral_start)
        return b
    
    def coeff_phi1(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0, 0, 0
            
        intAt_p1, intAt_p2 = self.coeff_intAt(t)
        def ode_fn3(t, y):
            a = self.get_a(t)
            return self.w * (1-t)**2 * torch.reciprocal(t) * torch.exp(a)
        def ode_fn4(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t)**2 * torch.reciprocal(t) * torch.exp(a) * (torch.exp(b) - 1)
        p3 = odeint(
            ode_fn3,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p4 = odeint(
            ode_fn4,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]

        return intAt_p1, intAt_p2, p3, p4
    
    def coeff_phi2(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return self.w * (1-t) * torch.reciprocal(t) * torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t) * torch.reciprocal(t) * torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2
    
    def coeff_phi3(self, t):
        if t - self.integral_start < 1e-6:
            return 0, 0
        def ode_fn1(t, y):
            a = self.get_a(t)
            return self.w * (1-t)**2 * torch.reciprocal(t) * torch.exp(a)
        def ode_fn2(t, y):
            a = self.get_a(t)
            b = self.get_b(t)
            return self.w * (1-t)**2 * torch.reciprocal(t) * torch.exp(a) * (torch.exp(b) - 1)
        p1 = odeint(
            ode_fn1,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        p2 = odeint(
            ode_fn2,
            self.init,
            torch.tensor([self.integral_start, t], device=self.device, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
            method="scipy_solver",
            options={"solver": "RK45"},
        )[-1]
        return p1, p2
    
    def get_coeff(self, n_steps):
        dt = (1 - self.st) / n_steps
        coeffs = []
        for i in torch.arange(n_steps + 1):
            t = (self.st + i * dt).clamp(max = 1)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4 = self.coeff_phi1(t)
            ph2_p1, ph2_p2 = self.coeff_phi2(t)
            ph3_p1, ph3_p2 = self.coeff_phi3(t)
            coeffs.append([ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2, ph3_p1, ph3_p2])
        return coeffs

    
    @torch.no_grad()
    def sample_conj(self, y, img_dim, n_steps=40):
        bs = y.shape[0]
        y1 = self.H_pinv(y)
        xt = self.st * y1 + (1 - self.st) * torch.randn_like(y1)
        xt_bar = self.At(xt, self.st)
        dt = (1 - self.st) / n_steps
        coeffs = self.get_coeff(n_steps)
        for i in tqdm(range(n_steps)):
            t = self.st + i * dt
            xt = self.At_inv(xt_bar, t)
            ph1_intAt_p1, ph1_intAt_p2, ph1_p3, ph1_p4, ph2_p1, ph2_p2, ph3_p1, ph3_p2 = coeffs[i]
            ph1_intAt_p1_plus, ph1_intAt_p2_plus, ph1_p3_plus, ph1_p4_plus, ph2_p1_plus, ph2_p2_plus, ph3_p1_plus, ph3_p2_plus = coeffs[i+1]
            t_tensor = t * torch.ones(bs, device=self.device)

            # jacobian term
            with torch.enable_grad():
                xt.requires_grad_(True)
                tmp_xt = xt.reshape(bs, *img_dim)
                vt = self.v_theta(tmp_xt, t_tensor * 999)
                vt = vt.reshape(bs, -1)
                tmp_xt = tmp_xt.reshape(bs, -1)
                pred_x1 = tmp_xt + (1 - t) * vt
                mat = (y1 - self.proj(pred_x1)).reshape(bs, -1)
                mat_x = (mat.detach() * vt).sum()
                # print(mat_x.requires_grad, vt.requires_grad, xt.requires_grad)
                grad_term = torch.autograd.grad(mat_x, xt)[0]
                # print(grad_term)
                xt.requires_grad_(False)

            dgrad_term = dt * (ph3_p1_plus - ph3_p1) * grad_term + dt * (ph3_p2_plus - ph3_p2) * grad_term

            # gradient term non-integral
            # dgrad_term = dt * self.At(grad_term, t, True) * self.w * (1-t)**2 * t

            # v_theta term
            proj_vt = self.proj(vt)
            dv = (ph1_intAt_p1_plus - ph1_intAt_p1) * vt + (ph1_intAt_p2_plus - ph1_intAt_p2) * proj_vt - (ph1_p3_plus - ph1_p3) * proj_vt - (ph1_p4_plus - ph1_p4) * proj_vt

            # v_theta term non-integral
            # dv = dt * (self.At(vt, t) - self.w * (1-t)**2 * t * self.At(self.proj(vt), t, True))

            # y term
            dy = (ph2_p1_plus - ph2_p1) * y1 + (ph2_p2_plus - ph2_p2) * y1

            # y term non-integral
            # dy = dt * self.w * (1-t)**2 * t * self.At(y1, t, True)

            # head term
            dhead = self.B * xt_bar * dt

            dxt_bar = dhead + dv + dy + dgrad_term
            xt_bar = (xt_bar + dxt_bar).float()
        xt = self.At_inv(xt_bar, torch.ones_like(self.st))
        return xt.reshape(bs, *img_dim)
    

class Unconditional:
    def __init__(self, model):
        self.model = model
        self.device = self.model.device
    @torch.no_grad()
    def sample(self, batch_image_shape, sample_steps=10):
        x = torch.randn(*batch_image_shape, device=self.device)
        dt = 1. / sample_steps
        eps = 1e-3
        for i in tqdm(range(sample_steps)):
            num_t = i / sample_steps * (1 - eps) + eps
            t = torch.ones(batch_image_shape[0], device=self.device) * num_t
            pred = self.model(x, t * 999)
            x = x + dt * pred
        return x