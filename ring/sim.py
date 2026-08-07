"""ODE simulation of the ring-attractor dynamical system.

``run_sim`` performs Euler integration of

    tau * dx/dt = -x + J @ phi(x) + omega(t) * J' @ phi(x) + stabilizer

on GPU/CPU torch tensors, with optional periodic save of state and an
early-stop check that fires when per-batch derivatives plateau.
"""
import numpy as np
import torch
from . import fn


def run_sim(T, tau, dt, T_eval, J, x_init,
            nonlin=fn.unity_erf,
            omega_func=None, J_prime=None,
            target_mean=None, const=np.nan,
            return_derivs=False,
            break_if_stopped=True,
            disable_tqdm=False):
    """Euler-integrate the ring-attractor ODE on GPU/CPU torch.

    The integrated equation is

        tau * dx/dt = -x + J @ phi(x) + omega(t) * J' @ phi(x) + stabilizer

    where ``phi`` is the nonlinearity, ``omega(t)`` is an optional time-
    varying scalar drift, ``J_prime`` is the skew-coupling to ``omega``,
    and the stabilizer (if enabled) clamps the population mean of
    ``phi(x)`` toward ``target_mean``.

    Parameters
    ----------
    T : float
        Total integration time.
    tau : float
        Membrane time constant (same units as ``T``, ``dt``).
    dt : float
        Integration step.
    T_eval : float or None
        Period at which to save state into ``X_save``. ``None`` disables saving.
    J : torch.Tensor, shape (N, N)
        Recurrent weight matrix.
    x_init : torch.Tensor, shape (N, N_batch)
        Initial state for ``N_batch`` parallel trajectories.
    nonlin : callable, default ``fn.unity_erf``
        Pointwise nonlinearity ``phi(x)``.
    omega_func : callable or None
        Function ``t -> omega(t)``. ``None`` disables the omega term.
    J_prime : torch.Tensor or None, shape (N, N)
        Required if ``omega_func`` is not None.
    target_mean : float or None
        If given, adds a stabilizer
        ``const * (target_mean - phi(x).mean(0))`` to the dynamics.
    const : float, default ``np.nan``
        Stabilizer strength; ignored when ``target_mean`` is None.
    return_derivs : bool, default False
        If True, also return per-batch RMS derivatives at exit.
    break_if_stopped : bool, default True
        If True, break out of the loop when all batch trajectories'
        RMS derivatives drop below 1e-100 (numerical zero).
    disable_tqdm : bool, default False
        Disable the tqdm progress bar (emits periodic plain-text status
        prints instead).

    Returns
    -------
    x_final : ndarray, shape (N, N_batch)
        Final state.
    X_save : ndarray or None, shape (N_eval, N, N_batch)
        Periodic snapshots of state; ``None`` when ``T_eval`` is None.
    finished : bool
        True if all ``N_t = int(T/dt)`` steps completed; False if the
        early-stop fired.
    derivs : ndarray, shape (N_batch,), optional
        Final per-batch RMS derivatives. Only returned if
        ``return_derivs`` is True.
    """
    device = J.device
    N_t = int(T / dt)
    N, N_batch = x_init.shape
    x = x_init
    x_prev = x.clone()
    check_deriv_iter = int(10 / dt)
    finished = True
    derivs = torch.zeros(N_batch, device=device)

    save = T_eval is not None
    if save:
        N_eval = int(T / T_eval)
        eval_iter = int(T_eval / dt)
        X_save = np.zeros((N_eval, N, N_batch))
        X_save[0] = x.cpu().numpy()

    if disable_tqdm:
        counter = range(1, N_t)
    else:
        from tqdm import tqdm
        counter = tqdm(range(1, N_t))

    for i in counter:
        r = nonlin(x)
        stabilizer = const * (target_mean - r.mean(0))[None, :] if target_mean is not None else 0.
        dxdt = (1. / tau) * (-x + torch.mm(J, r) + stabilizer)
        if omega_func is not None:
            dxdt += (1. / tau) * tau * omega_func(i * dt) * torch.mm(J_prime, r)
        x += dt * dxdt

        if save and i % eval_iter == 0:
            X_save[i // eval_iter] = x.cpu().numpy()

        if i % check_deriv_iter == 0:
            diff = x - x_prev
            derivs = torch.sqrt((diff**2).mean(0))
            if derivs.max().item() <= 1e-100:
                finished = False
                if break_if_stopped:
                    break
            x_prev = x.clone()

        if disable_tqdm and i % 100000 == 0:
            print(f"t = {i*dt}", flush=True)

    x_final = x.cpu().numpy()
    X_save = X_save if save else None

    if return_derivs:
        return x_final, X_save, finished, derivs.cpu().numpy()
    return x_final, X_save, finished
