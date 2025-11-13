import jax
import jax.numpy as jnp

from typing import Literal, Optional, Tuple


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


class MiniLinear:
    def __init__(self, in_features: int, out_features: int, init_kind: Literal["xavier", "he"] = "xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.init_kind = init_kind
    
    def __call__(self, W: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """dimensions wise: (B, D) @ (D, H) + (H,) = (B, H)"""
        return x @ W + b

    def init_params(self, key: jax.random.PRNGKey) -> dict:
        w_key, b_key = jax.random.split(key)
        W = self.initialize_weights(w_key, kind=self.init_kind)
        b = self.initialize_bias(b_key)
        
        return {"W": W, "b": b}

    def initialize_weights(self, key: jax.random.PRNGKey, kind: Literal["xavier", "he"] = "xavier") -> jnp.ndarray:
        """Initializes the weights of the layer using either Xavier or He initialization."""
        if kind == "xavier":
            return jax.random.normal(key, (self.in_features, self.out_features)) * jnp.sqrt(1 / self.in_features)
        elif kind == "he":
            return jax.random.normal(key, (self.in_features, self.out_features)) * jnp.sqrt(2 / self.in_features)

    def initialize_bias(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Initializes the bias of the layer."""
        return jnp.zeros((self.out_features,))


class MiniMLP:
    def __init__(self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        n_layers: int = 3,
        init_kind: Literal["xavier", "he"] = "he",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_layers = n_layers
        self.init_kind = init_kind

        self.layers = []
        if n_layers == 1:
            self.layers.append(MiniLinear(in_features, out_features, init_kind))
        else:
            self.layers.append(MiniLinear(in_features, hidden_features, init_kind))
            for _ in range(n_layers - 2):
                self.layers.append(MiniLinear(hidden_features, hidden_features, init_kind))
            
            self.layers.append(MiniLinear(hidden_features, out_features, init_kind))

    def __call__(self, params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(params, x)

    def init_params(self, key: jax.random.PRNGKey) -> list[dict]:
        layer_keys = jax.random.split(key, self.n_layers)
        return [layer.init_params(k) for layer, k in zip(self.layers, layer_keys)]

    def forward(self, params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
        # Apply hidden layers with ReLU activations
        for layer, p in zip(self.layers[:-1], params[:-1]):
            x = layer(**p, x=x)
            x = relu(x)
        
        # Apply the final layer without activation
        x = self.layers[-1](**params[-1], x=x)
        
        return x

if __name__ == "__main__":
    import time
    key = jax.random.PRNGKey(42)
    mlp = MiniMLP(in_features=10, out_features=2, hidden_features=512, n_layers=10)
    params = mlp.init_params(key)
    x = jnp.zeros((100, 10))  # (B, D)
    learning_rate = 1e-2
    
    start = time.perf_counter()
    output = mlp(params, x)
    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    start = time.perf_counter()
    mlp_forward = jax.jit(mlp.forward)
    output = mlp_forward(params, x)
    end = time.perf_counter()
    print(f"Time taken (jit): {end - start} seconds")

    ground_truth = jnp.zeros((100, 2))

    start = time.perf_counter()
    loss_function = lambda params: ((mlp_forward(params, x) - ground_truth)**2).mean()
    grad_function = jax.grad(loss_function)
    grads = grad_function(params)
    for param, grad in zip(params, grads):
        param["W"] -= learning_rate * grad["W"]
        param["b"] -= learning_rate * grad["b"]
    
    end = time.perf_counter()
    print(f"Time taken (naive grad): {end - start} seconds")

    start = time.perf_counter()    
    loss_and_grad_fn = jax.value_and_grad(lambda p: ((mlp_forward(p, x) - ground_truth)**2).mean())
    @jax.jit
    def train_step(params):
        loss, grads = loss_and_grad_fn(params)
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
        return params, loss

    params, loss_val = train_step(params)
    end = time.perf_counter()
    print(f"Time taken (train step): {end - start} seconds")
