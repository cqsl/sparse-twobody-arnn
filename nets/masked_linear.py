from math import sqrt
from typing import Optional

import equinox as eqx
import jax
from jax import numpy as jnp
from jax._src.typing import DType
from jaxtyping import Array, PRNGKeyArray


class MaskedDense1D(eqx.Module):
    kernel: Array
    bias: Optional[Array]
    N: int = eqx.field(static=True)
    n_ham_params: int = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    exclusive: bool = eqx.field(static=True)
    param_dtype: DType = eqx.field(static=True)

    def __init__(
        self,
        N: int,
        n_ham_params: int,
        in_features: int,
        out_features: int,
        *,
        exclusive: bool,
        use_bias: bool = True,
        param_dtype: DType = None,
        key: PRNGKeyArray = None,
        small_init: bool = False,
    ):
        self.N = N
        self.n_ham_params = n_ham_params
        self.in_features = in_features
        self.out_features = out_features
        self.exclusive = exclusive
        self.param_dtype = param_dtype

        key_k, key_b = jax.random.split(key)

        scale = 1 / sqrt(N * in_features)
        # Last layer has small weights,
        # so the wave function is close to uniform initially
        if small_init:
            scale *= 1e-2

        in_size = N * in_features + n_ham_params
        mask = self.get_mask()
        self.kernel = (
            mask
            * scale
            * jax.random.truncated_normal(
                key_k, -2, 2, (in_size, N * out_features), param_dtype
            )
        )
        if use_bias:
            self.bias = scale * jax.random.truncated_normal(
                key_b, -2, 2, (N * out_features,), param_dtype
            )
        else:
            self.bias = None

    def get_mask(self):
        with jax.ensure_compile_time_eval():
            N = self.N
            dtype = self.param_dtype

            mask = jnp.ones((N, N), dtype=dtype)
            mask = jnp.triu(mask, self.exclusive)
            features = jnp.ones((self.in_features, self.out_features), dtype=dtype)
            mask = jnp.kron(mask, features)
            pad = jnp.ones((self.n_ham_params, N * self.out_features), dtype=dtype)
            mask = jnp.concatenate([mask, pad], axis=0)
            return mask

    def __call__(self, x: Array, ham_params: Array) -> Array:
        N = self.N
        B = x.shape[0]

        x = x.reshape((B, N * self.in_features))
        if self.n_ham_params > 0:
            ham_params = jnp.broadcast_to(ham_params, (B, self.n_ham_params))
            x = jnp.concatenate([x, ham_params], axis=1)

        mask = self.get_mask()
        y = x @ (mask * self.kernel)

        if self.bias is not None:
            y += self.bias

        y = y.reshape((B, N, self.out_features))
        return y
