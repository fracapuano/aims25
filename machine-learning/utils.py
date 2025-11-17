from typing import TypeAlias

import jax.numpy as jnp

ModelParameters: TypeAlias = dict[str, dict[str, jnp.ndarray]]
Gradients: TypeAlias = dict[str, dict[str, jnp.ndarray]]
OptimizerState: TypeAlias = dict[str, dict[str, jnp.ndarray]]


def orthogonalize(M: jnp.ndarray) -> jnp.ndarray:
    "from https://docs.modula.systems/algorithms/newton-schulz/"
    assert M.ndim == 2, "Orthogonalization is implemented for 2D tensors only!"
    
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / jnp.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    
    if transpose:
        M = M.T
    
    return M

def cosine_scheduling(warmup_steps: int, total_steps: int, peak_lr: float):
    """A minimalistic implementation of cosine scheduling with linear warmup."""
    assert total_steps != warmup_steps, "Warmup steps must be strictly less than total steps!"
    decay_steps = total_steps - warmup_steps

    def _lr(step: int)->float:
        step = jnp.asarray(step)
        warmup_lr = peak_lr * (step / warmup_steps)
        t = jnp.clip((step - warmup_steps) / decay_steps, 0., 1.)

        cosine_lr = 0.5 * peak_lr * (1+jnp.cos(jnp.pi * t))
        return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)

    return _lr