from typing import TypeAlias

import jax.numpy as jnp


ModelParameters: TypeAlias = dict[str, dict[str, jnp.ndarray]]
Gradients: TypeAlias = dict[str, dict[str, jnp.ndarray]]
OptimizerState: TypeAlias = dict[str, dict[str, jnp.ndarray]]