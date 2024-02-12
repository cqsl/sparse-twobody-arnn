# x1 in the code is \xi in the paper

from math import sqrt
from typing import Callable

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax._src.typing import DType
from jaxtyping import Array, PRNGKeyArray

from .arnn import AbstractARNN

vmap_matmul = jax.vmap(jnp.matmul, in_axes=(1, 0), out_axes=1)


def vmap_take(J, mask):
    return jnp.take(J, mask, axis=1).transpose((1, 0, 2))


class AbstractTwoBo(AbstractARNN):
    def _get_mask_J(self, J):
        N = J.shape[0]
        mask_J = []
        len_x1 = 0
        for idx in range(N):
            # Select the non-zero rows of the interaction matrix J
            sub_J = J[:idx, idx:]
            non_zero_cols = np.any(sub_J != 0, axis=0)
            indices = np.argwhere(non_zero_cols).squeeze(axis=1)
            mask_J.append(idx + indices)
            len_x1 = max(len_x1, len(indices))
        for idx in range(N):
            mask_J[idx] = np.pad(mask_J[idx], ((0, len_x1 - mask_J[idx].size),))
        mask_J = np.stack(mask_J)
        # mask_J: (N, len_x1), mask_J[0, :] == 0
        mask_J = jnp.asarray(mask_J)
        return mask_J

    def __call__(self, x: Array, ham_params: Array, beta: float) -> Array:
        x_hat = self.conditionals(x, ham_params, beta)
        log_p = self.get_log_p(x, x_hat)
        return log_p

    # idx in [0, N - 1]
    # Returns: (batch_size,)
    def conditional(self, x: Array, idx: int, J: Array, beta: float) -> Array:
        if idx == 0:
            batch_size = x.shape[0]
            x_hat_0 = jax.nn.sigmoid(self.b0)
            x_hat_0 = jnp.broadcast_to(x_hat_0, (batch_size,))
            return x_hat_0
        else:
            x1_i = self.first_layer_i(x, idx, J, beta, self.use_beta_skip)
            x_hat_i = self.rest_layers_i(x1_i, idx)
            return x_hat_i

    # Returns: (batch_size, N)
    def conditionals(self, x: Array, J: Array, beta: float) -> Array:
        batch_size = x.shape[0]

        x1 = self.first_layer(x, J, beta, self.use_beta_skip)
        x_hat = self.rest_layers(x1)

        x_hat_0 = jax.nn.sigmoid(self.b0)
        x_hat_0 = jnp.broadcast_to(x_hat_0, (batch_size, 1))
        x_hat = jnp.concatenate([x_hat_0, x_hat], axis=1)
        return x_hat

    def sample(
        self,
        batch_size: int,
        N: int,
        ham_params: Array,
        beta: float,
        key: PRNGKeyArray,
    ) -> Array:
        J = ham_params
        keys = jax.random.split(key, N)

        def scan_fun(carry, idx):
            x, x_hat = carry
            x1_i = self.first_layer_i(x, idx, J, beta, self.use_beta_skip)
            x_hat_i = self.rest_layers_i(x1_i, idx)
            x_hat = x_hat.at[:, idx].set(x_hat_i)
            x_i = jax.random.bernoulli(keys[idx], x_hat_i).astype(x.dtype) * 2 - 1
            x = x.at[:, idx].set(x_i)
            return (x, x_hat), None

        x = jnp.zeros((batch_size, N), dtype=ham_params.dtype)
        x_hat = jnp.zeros_like(x)
        x_hat = x_hat.at[:, 0].set(jax.nn.sigmoid(self.b0))
        x = x.at[:, 0].set(
            jax.random.bernoulli(keys[0], x_hat[:, 0]).astype(x.dtype) * 2 - 1
        )
        (x, x_hat), _ = jax.lax.scan(scan_fun, (x, x_hat), jnp.arange(1, N))
        return x, x_hat

    # Returns: (batch_size, len_x1)
    def first_layer_i(
        self, x: Array, idx: int, J: Array, beta: float, use_beta_skip: bool
    ) -> Array:
        x_i = x * self.mask_x[idx]
        J_i = jnp.take(J, self.mask_J[idx], axis=1)
        x1_i = x_i @ J_i

        x1_i = x1_i.at[:, 0].multiply(jnp.where(use_beta_skip, beta, 1))
        return x1_i

    # Returns: (batch_size, N - 1, len_x1)
    def first_layer(
        self, x: Array, J: Array, beta: float, use_beta_skip: bool
    ) -> Array:
        # Use scan rather than vmap to reduce memory usage
        def scan_fun(_, idx):
            x_i = x * self.mask_x[idx]
            J_i = jnp.take(J, self.mask_J[idx], axis=1)
            x1_i = x_i @ J_i
            return None, x1_i

        N = x.shape[1]
        _, x1 = jax.lax.scan(scan_fun, None, jnp.arange(1, N))
        x1 = x1.transpose((1, 0, 2))

        x1 = x1.at[:, :, 0].multiply(jnp.where(use_beta_skip, beta, 1))
        return x1

    def rest_layers_i(self, x1: Array, idx: int) -> Array:
        raise NotImplementedError

    def rest_layers(self, x1: Array) -> Array:
        raise NotImplementedError


class TwoBo(AbstractTwoBo):
    w: list[Array]
    b: list[Array]
    b0: Array
    w_skip: Array
    b_skip: Array
    activation: Callable[[Array], Array] = eqx.field(static=True)
    param_dtype: DType = eqx.field(static=True)
    use_beta_skip: bool = eqx.field(static=True)
    mask_x: Array = eqx.field(static=True)
    mask_J: Array = eqx.field(static=True)
    len_x1: int = eqx.field(static=True)

    def __init__(
        self,
        N: int,
        # We assume that J == np.triu(J + J.T, k=1)
        J: Array,
        layers_shape: list[int],
        *,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        param_dtype: DType = None,
        key: PRNGKeyArray = None,
        # The factor 2 in 2 \beta \xi_ii can be specified in weight_skip
        weight_skip: float = 2,
        # Whether anneal the \beta in 2 \beta \xi_ii
        use_beta_skip: bool = True,
    ):
        self.activation = activation
        self.param_dtype = param_dtype
        self.use_beta_skip = use_beta_skip

        mask_x = np.tril(np.ones((N, N), dtype=param_dtype), k=-1)
        self.mask_x = jnp.asarray(mask_x)

        self.mask_J = self._get_mask_J(J)
        len_x1 = self.mask_J.shape[1]
        self.len_x1 = len_x1

        # Initialize parameters
        n_layers = len(layers_shape)
        key_w, key_b, key_b0, key_w_skip, key_b_skip = jax.random.split(key, 5)
        keys_w = jax.random.split(key_w, n_layers)
        keys_b = jax.random.split(key_b, n_layers)

        # Last layer has small weights,
        # so the wave function is close to uniform initially
        small_scale = 1e-2 / sqrt(N)
        self.w = []
        self.b = []
        for i in range(n_layers):
            in_features = len_x1 - 1 if i == 0 else layers_shape[i - 1]
            out_features = layers_shape[i]
            scale = small_scale if i == n_layers - 1 else 1 / sqrt(in_features)

            w = scale * jax.random.truncated_normal(
                keys_w[i], -2, 2, (N - 1, in_features, out_features), param_dtype
            )
            if i == 0:
                # Mask out unused parameters
                w = jnp.where(self.mask_J[1:, 1:, None] == 0, 0, w)
            self.w.append(w)

            b = scale * jax.random.truncated_normal(
                keys_b[i], -2, 2, (N - 1, out_features), param_dtype
            )
            self.b.append(b)

        self.b0 = small_scale * jax.random.truncated_normal(
            key_b0, -2, 2, (), param_dtype
        )

        w_skip = small_scale * jax.random.truncated_normal(
            key_w_skip, -2, 2, (N - 1, len_x1, 1), param_dtype
        )
        w_skip = w_skip.at[:, 0, 0].add(weight_skip)
        # Mask out unused parameters
        w_skip = jnp.where(self.mask_J[1:, :, None] == 0, 0, w_skip)
        self.w_skip = w_skip

        self.b_skip = small_scale * jax.random.truncated_normal(
            key_b_skip, -2, 2, (N - 1, 1), param_dtype
        )

    # Here x1 is actually x1_i
    # idx in [1, N - 1]
    # Returns: (batch_size,)
    def rest_layers_i(self, x1: Array, idx: int) -> Array:
        x1_skip = x1 @ self.w_skip[idx - 1] + self.b_skip[idx - 1]

        # \rho_i only depends on \xi_{i l | l > i}
        x1 = x1[:, 1:]
        for _w, _b in zip(self.w, self.b):
            x1 = x1 @ _w[idx - 1] + _b[idx - 1]
            x1 = self.activation(x1)

        x1 = jnp.squeeze(x1_skip + x1, axis=-1)
        return jax.nn.sigmoid(x1)

    # Returns: (batch_size, N - 1)
    def rest_layers(self, x1: Array) -> Array:
        x1_skip = vmap_matmul(x1, self.w_skip) + self.b_skip

        # \rho_i only depends on \xi_{i l | l > i}
        x1 = x1[:, :, 1:]
        for _w, _b in zip(self.w, self.b):
            x1 = vmap_matmul(x1, _w) + _b
            x1 = self.activation(x1)

        x1 = jnp.squeeze(x1_skip + x1, axis=-1)
        return jax.nn.sigmoid(x1)


# A simpler version of TwoBo that uses only a single dense layer to parameterize
# each \rho_i
class TwoBoOnlySkip(AbstractTwoBo):
    b0: Array
    w_skip: Array
    b_skip: Array
    activation: Callable[[Array], Array] = eqx.field(static=True)
    param_dtype: DType = eqx.field(static=True)
    use_beta_skip: bool = eqx.field(static=True)
    mask_x: Array = eqx.field(static=True)
    mask_J: Array = eqx.field(static=True)
    len_x1: int = eqx.field(static=True)

    def __init__(
        self,
        N: int,
        # We assume that J == np.triu(J + J.T, k=1)
        J: Array,
        *,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        param_dtype: DType = None,
        key: PRNGKeyArray = None,
        # The factor 2 in 2 \beta \xi_ii can be specified in weight_skip
        weight_skip: float = 2,
        # Whether anneal the \beta in 2 \beta \xi_ii
        use_beta_skip: bool = True,
    ):
        self.activation = activation
        self.param_dtype = param_dtype
        self.use_beta_skip = use_beta_skip

        mask_x = np.tril(np.ones((N, N), dtype=param_dtype), k=-1)
        self.mask_x = jnp.asarray(mask_x)

        self.mask_J = self._get_mask_J(J)
        len_x1 = self.mask_J.shape[1]
        self.len_x1 = len_x1

        # Initialize parameters
        key_b0, key_w_skip, key_b_skip = jax.random.split(key, 3)

        # Last layer has small weights,
        # so the wave function is close to uniform initially
        small_scale = 1e-2 / sqrt(N)
        self.b0 = small_scale * jax.random.truncated_normal(
            key_b0, -2, 2, (), param_dtype
        )

        w_skip = small_scale * jax.random.truncated_normal(
            key_w_skip, -2, 2, (N - 1, len_x1, 1), param_dtype
        )
        w_skip = w_skip.at[:, 0, 0].add(weight_skip)
        # Mask out unused parameters
        w_skip = jnp.where(self.mask_J[1:, :, None] == 0, 0, w_skip)
        self.w_skip = w_skip

        self.b_skip = small_scale * jax.random.truncated_normal(
            key_b_skip, -2, 2, (N - 1, 1), param_dtype
        )

    # Here x1 is actually x1_i
    # idx in [1, N - 1]
    # Returns: (batch_size,)
    def rest_layers_i(self, x1: Array, idx: int) -> Array:
        x1_skip = x1 @ self.w_skip[idx - 1] + self.b_skip[idx - 1]
        x1 = jnp.squeeze(x1_skip, axis=-1)
        return jax.nn.sigmoid(x1)

    # Returns: (batch_size, N - 1)
    def rest_layers(self, x1: Array) -> Array:
        x1_skip = vmap_matmul(x1, self.w_skip) + self.b_skip
        x1 = jnp.squeeze(x1_skip, axis=-1)
        return jax.nn.sigmoid(x1)
