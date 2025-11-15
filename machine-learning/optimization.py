from typing import Optional
from utils import ModelParameters, Gradients, OptimizerState

import jax
import jax.numpy as jnp
import wandb


def sgd_update(params: ModelParameters, grad: Gradients, learning_rate: float) -> OptimizerState:
    return {
        "params": jax.tree.map(lambda p, g: p - learning_rate * g, params, grad)
    }

def momentum_update(params: ModelParameters, grad: Gradients, momentum: Gradients, beta: float, learning_rate: float) -> OptimizerState:
    momentum_updated = jax.tree.map(
        lambda m, g: beta * m + (1 - beta) * g, momentum, grad
    )

    return {
        "params": jax.tree.map(lambda p, m: p - learning_rate * m, params, momentum_updated),
        # Optimizer state
        "momentum": momentum_updated,
        "beta": beta
    }

def adagrad_update():
    pass

def rmsprop_update():
    pass

def adam_update():
    pass

def adamw_update():
    pass

def _2d_muon_update():
    pass

def muon_update():
    pass

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


UPDATES = {
    "sgd": sgd_update,
    "momentum": momentum_update,
    "adagrad": adagrad_update,
    "rmsprop": rmsprop_update,
    "adam": adam_update,
    "adamw": adamw_update,
    "muon": muon_update
}

PREPARE = {
    "sgd": lambda s: s,
    "momentum": lambda s: {"momentum": s["momentum"], "beta": s["beta"]},
}

def _initialize_momentum(params: ModelParameters, beta: float) -> ModelParameters:
    return {"momentum": jax.tree.map(jnp.zeros_like, params), "beta": beta}


class MiniOptimizer:
    def __init__(self, name:str, total_steps:int, warmup_steps: Optional[int]=None, peak_lr:Optional[float]=1e-3, kwargs:dict={}):  # or 3e-4 for Karpathy's constant
        if name not in UPDATES:
            raise ValueError(f"{name} must be an optimizer name. Currently supporting: {UPDATES.keys()}")
        
        self.total_steps = total_steps
        self.name = name

        self.kwargs = kwargs

        self._update_function = UPDATES[self.name]

        # defaults to 10% of training steps for warmup when not provided
        warmup_steps = total_steps // 10 if warmup_steps is None else warmup_steps
        self.scheduler = cosine_scheduling(warmup_steps, total_steps, peak_lr)
    
    def initialize_state(self, params: ModelParameters) -> OptimizerState:
        if self.name == "sgd":
            return {}
        elif self.name == "momentum":
            return _initialize_momentum(params=params, beta=self.kwargs["beta"])

    def prepare(self, state: OptimizerState) -> OptimizerState:
        return PREPARE[self.name](state)

    def update(self, params: ModelParameters, grad: Gradients, step: int, **kwargs) -> OptimizerState:
        learning_rate = self.scheduler(step)
        wandb.log({"train/lr": learning_rate})
        
        return self._update_function(params=params, grad=grad,learning_rate=learning_rate, **kwargs)


    # def update(self, params):
    #     for k, update in grad:
    #         if update.ndim ... :  # update for the weight *matrices*
    #             muon_update()
    #         else:
    #             adamw_update()

    #     return new_params