import jax
import jax.numpy as jnp

from network import MiniMLP, MiniLinear
from typing import Tuple, Optional, Literal

from copy import deepcopy


class MiniFlowMLP:
    def __init__(self, mlp: MiniMLP, n_steps: int = 10, mlp_time: Optional[MiniMLP] = None):
        self.mlp_velocity = mlp
        self.mlp_time = deepcopy(mlp) if mlp_time is None else mlp_time
        
        # the time mlp is embedding the (B, 1) tensor of time instants
        first_layer_shape = (self.mlp_velocity.layers[0].in_features, self.mlp_velocity.layers[0].out_features)
        self.mlp_time.layers[0] = MiniLinear(
            1,  # embedding batched scalar timestamps
            first_layer_shape[-1]
        )

        # jit the forward passes of both MLPs
        self.mlp_velocity_forward = jax.jit(self.mlp_velocity.forward)
        self.mlp_time_forward = jax.jit(self.mlp_time.forward)
        
        # number of integration steps (used for inference)
        self.n_steps = n_steps
    
    def init_params(self, key: jax.random.PRNGKey, kind: Literal["xavier", "he"] = "he") -> dict[str, jnp.ndarray]:
        mlp_velocity_params = self.mlp_velocity.init_params(key)
        mlp_time_params = self.mlp_time.init_params(key)

        params_dict = {
            "mlp_velocity": mlp_velocity_params,
            "mlp_time" : mlp_time_params
        }
        return params_dict

    def sample_noise(self, key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(key, shape)

    def sample_time(self, key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        return jax.random.uniform(key, (n_samples, 1))

    def forward(self, params: dict[str, dict[str, jnp.ndarray]], key: jax.random.PRNGKey, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forwards a batch of data x of shape (B, D) through the Flow Network."""
        noise_key, time_key = jax.random.split(key)
        noise = self.sample_noise(noise_key, x.shape)
        # sampling a single timestep per item in the batch
        time = self.sample_time(time_key, x.shape[0])

        # interpolates between observed data and pure random noise
        x_t = time * x + (1 - time) * noise
        u_t = noise - x

        time_embedding = self.mlp_time_forward(params["mlp_time"], time)

        flow_input = jnp.hstack((x_t, time_embedding))
        v_t = self.mlp_velocity_forward(params["mlp_velocity"], flow_input)

        loss = ((v_t - u_t)**2).mean()
        return loss, v_t

    def _velocity(self, params: dict, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        if t.ndim == 0 or isinstance(t, float):
            t = jnp.array([t])
        if t.ndim == 1:
            t = t[:, None]
        
        t = t * jnp.ones((x_t.shape[0], 1))

        time_embedding = self.mlp_time_forward(params["mlp_time"], t)
        velocity_mlp_input = jnp.hstack((x_t, time_embedding))
        
        v_t = self.mlp_velocity_forward(params["mlp_velocity"], velocity_mlp_input)

        return v_t

    def generate(self, params: dict, x0: jnp.ndarray) -> jnp.ndarray:
        """Generates a sample of an unknown distribution starting from an
        easy-to-sample distribution using flow matching."""
        dt = 1.0 / self.n_steps
        x_t = x0

        for i in range(self.n_steps):
            t = jnp.array(i * dt)
            v_t = self._velocity(params, x_t, t)
            x_t += dt * v_t

        return x_t
    
    #TODO(fracapuano): add a jax.lax.scan version of generation?

if __name__ == "__main__":
    import time
    key = jax.random.PRNGKey(42)
    key, mlp_key, flow_key, data_key = jax.random.split(key, 4)

    # Mapping a 2D distribution space to another 2D distribution
    BATCH_SIZE = 32
    DIM_IN = 2
    DIM_OUT = 2

    mlp_config = dict(
        # stacking data and time embeddings
        in_features=DIM_IN * 2,
        out_features=DIM_OUT
    )

    flow_config = dict(
        n_steps=10
    )

    mlp = MiniMLP(**mlp_config)
    params = mlp.init_params(mlp_key)
    flowmlp = MiniFlowMLP(mlp, **flow_config)

    params = flowmlp.init_params(flow_key)

    x = jax.random.normal(data_key, (BATCH_SIZE, DIM_IN))

    print("Running warm-up calls for JIT compilation...")
    flowmlp_forward_jit = jax.jit(flowmlp.forward)
    flowmlp_generate_jit = jax.jit(flowmlp.generate)

    key, warmup_key = jax.random.split(key)
    x_warmup = jax.random.normal(warmup_key, x.shape)

    loss_warmup, v_t_warmup = flowmlp_forward_jit(params, key, x)
    loss_warmup.block_until_ready()  # block until ready useful to stop async execution
    flowmlp_generate_jit(params, x_warmup).block_until_ready()
    print("Warm-up complete.")

    start = time.perf_counter()
    loss, v_t = flowmlp.forward(params, key, x)
    # No block_until_ready needed for non-JIT on CPU, but good practice
    end = time.perf_counter()
    print(f"Forward time (non-jit): {end - start:.6f} seconds")

    start = time.perf_counter()
    loss, v_t = flowmlp_forward_jit(params, key, x)
    loss.block_until_ready()
    end = time.perf_counter()
    print(f"Forward time (jit): {end - start:.6f} seconds")

    key, generation_key = jax.random.split(key)
    x0 = jax.random.normal(generation_key, x.shape)

    start = time.perf_counter()
    generated_sample = flowmlp.generate(params, x0)
    end = time.perf_counter()
    print(f"Sample generation time (non-jit, python loop): {end - start:.6f} seconds")

    start = time.perf_counter()
    generated_sample = flowmlp_generate_jit(params, x0)
    generated_sample.block_until_ready()
    end = time.perf_counter()
    print(f"Sample generation time (jit, python loop): {end - start:.6f} seconds")
