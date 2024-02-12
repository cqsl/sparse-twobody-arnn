from typing import Callable

import equinox as eqx
import jax
from jax import numpy as jnp
from jax._src.typing import DType
from jaxtyping import Array, PRNGKeyArray

from .masked_linear import MaskedDense1D


class AbstractARNN(eqx.Module):
    def conditionals(self, x: Array, ham_params: Array, beta: float) -> Array:
        raise NotImplementedError

    def conditional(self, x: Array, idx: int, ham_params: Array, beta: float) -> Array:
        return self.conditionals(x, ham_params, beta)[:, idx]

    def get_log_p(self, x: Array, x_hat: Array, eps: float = 1e-7) -> Array:
        mask = (x + 1) / 2
        log_p = jnp.log(x_hat + eps) * mask
        log_p += jnp.log(1 - x_hat + eps) * (1 - mask)
        log_p = log_p.sum(axis=1)
        return log_p

    def __call__(self, x: Array, ham_params: Array, beta: float) -> Array:
        x_hat = self.conditionals(x, ham_params, beta)
        log_p = self.get_log_p(x, x_hat)
        return log_p

    def sample(
        self, batch_size: int, N: int, ham_params: Array, beta: float, key: PRNGKeyArray
    ) -> Array:
        keys = jax.random.split(key, N)

        def scan_fun(carry, idx):
            x, x_hat = carry
            new_x_hat = self.conditional(x, idx, ham_params, beta)
            new_x = jax.random.bernoulli(keys[idx], new_x_hat).astype(x.dtype) * 2 - 1
            x = x.at[:, idx].set(new_x)
            x_hat = x_hat.at[:, idx].set(new_x_hat)
            return (x, x_hat), None

        x = jnp.zeros((batch_size, N), dtype=ham_params.dtype)
        x_hat = jnp.zeros_like(x)
        (x, x_hat), _ = jax.lax.scan(scan_fun, (x, x_hat), jnp.arange(N))
        return x, x_hat


class ARNNSequential(AbstractARNN):
    layers: list[eqx.Module]
    activation: Callable[[Array], Array] = eqx.field(static=True)

    # beta is not used
    def conditionals(self, x: Array, ham_params: Array, beta: float) -> Array:
        m, N = x.shape

        x = self.reshape_inputs(x)
        x = jnp.expand_dims(x, axis=-1)

        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.activation(x)
            x = layer(x, ham_params)

        x = jax.nn.sigmoid(x)
        x = x.reshape((m, N))
        return x

    def reshape_inputs(self, x: Array) -> Array:
        return x


class ARNNDense(ARNNSequential):
    def __init__(
        self,
        N: int,
        n_ham_params: int,
        layers: int,
        features: int,
        *,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        param_dtype: DType = None,
        key: PRNGKeyArray = None,
    ):
        _features = [1] + [features] * (layers - 1) + [1]
        keys = jax.random.split(key, layers)
        self.layers = [
            MaskedDense1D(
                N,
                n_ham_params,
                in_features=_features[i],
                out_features=_features[i + 1],
                exclusive=(i == 0),
                param_dtype=param_dtype,
                key=keys[i],
                small_init=(i == layers - 1),
            )
            for i in range(layers)
        ]
        self.activation = activation
