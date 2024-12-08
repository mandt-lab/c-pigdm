from torchdiffeq import odeint
import torch


B=1e-8*torch.ones(1)
integral_start=1e-2
t=torch.ones(1)*0.5

def get_a(t):
    a = B * (t - 0)
    return a

def fn1(t):
    a = get_a(t)
    if B == 0:
        return 0
    return torch.exp(a) / B

def ode_fn1(t, y):
    a = get_a(t)
    return torch.exp(a)

p1 = odeint(
        ode_fn1,
        torch.zeros(1),
        torch.tensor([0, t], dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
        method="scipy_solver",
        options={"solver": "RK45"},
    )[-1]

print(p1)
print(fn1(t)-fn1(0))