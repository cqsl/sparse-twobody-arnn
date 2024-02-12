from functools import partial
from typing import Optional

import equinox as eqx
import jax
from jax import numpy as jnp
from jax._src.typing import DType
from jax.scipy.special import logsumexp
from jaxtyping import Array, PRNGKeyArray


class AbstractSpinsModel(eqx.Module):
    J: Array
    h: Optional[Array]
    N: int = eqx.field(static=True)
    dtype: DType = eqx.field(static=True)

    @property
    def n_params(self) -> int:
        raise NotImplementedError

    @property
    def flat_params(self) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array) -> Array:
        """
        Compute energies of samples x.

        x should be of shape (m, N),
        where m is the number of samples, and N is the number of spins.
        """
        raise NotImplementedError


def numbers_to_states(x: Array, n_bits: int, dtype: DType) -> Array:
    mask = 2 ** jnp.arange(n_bits, dtype=x.dtype)
    x = (x[:, None] & mask) > 0
    x = x.astype(dtype) * 2 - 1
    return x


def states_to_numbers(x: Array) -> Array:
    n_bits = x.shape[-1]
    x = (x > 0).astype(jnp.uint32)
    mask = 2 ** jnp.arange(n_bits, dtype=x.dtype)
    x @= mask
    return x


@partial(jax.jit, static_argnames=["chunk_size"])
def exact(
    ham: AbstractSpinsModel, beta: float, *, chunk_size: int = 65536
) -> dict[str, Array]:
    N = ham.N
    dtype = ham.dtype

    n_all = 2**N
    assert N <= 32

    confs_all = jnp.arange(n_all, dtype=jnp.uint32)
    if n_all > chunk_size:
        confs_all = jnp.split(confs_all, n_all // chunk_size)
    else:
        confs_all = confs_all[None, :]

    Es_all = []
    M_abs_sum = 0

    for confs_chunk in confs_all:
        x = numbers_to_states(confs_chunk, N, dtype)
        Es = ham(x)
        Es_all.append(Es)

        Z_chunk = jnp.exp(-beta * Es)
        M_abs_sum += (abs(x.mean(axis=1)) * Z_chunk).sum()

    Es_all = jnp.concatenate(Es_all)
    log_Z = logsumexp(-beta * Es_all)
    Z = jnp.exp(log_Z)

    free_energy = -1 / beta * log_Z / N
    energy = jnp.exp(-beta * Es_all + jnp.log(abs(Es_all)) - log_Z)
    energy = jnp.sign(Es_all) @ energy / N
    entropy = beta * (energy - free_energy)
    mag_abs = M_abs_sum / Z

    return {
        "free_energy": free_energy,
        "energy": energy,
        "entropy": entropy,
        "|M|": mag_abs,
    }


def batched_flatten(a: Array) -> Array:
    return a.reshape(a.shape[0], -1)


class GeneralSpinsModel(AbstractSpinsModel):
    def __init__(
        self,
        batch_size: int,
        N: int,
        *,
        J: Optional[Array] = None,
        h: Optional[Array] = None,
        dtype: Optional[DType] = None,
        key: Optional[PRNGKeyArray] = None,
    ):
        if dtype is None:
            assert J is not None
            assert h is not None
            dtype = jnp.promote_types(J.dtype, h.dtype)

        if key is None:
            assert J is not None
            assert h is not None
        else:
            assert J is None
            assert h is None
            key_J, key_h = jax.random.split(key)
            J = jax.random.normal(key_J, (batch_size, N, N), dtype)
            h = jax.random.normal(key_h, (batch_size, N), dtype)

        self.N = N
        self.J = J
        self.h = h
        self.dtype = dtype

    @property
    def n_params(self) -> int:
        return self.J[0].size + self.h[0].size

    @property
    def flat_params(self) -> Array:
        return jnp.concatenate(
            [batched_flatten(self.J), batched_flatten(self.h)], axis=1
        )

    def __call__(self, x: Array) -> Array:
        J = self.J
        h = self.h
        m, N = x.shape

        inter = -jnp.einsum("mij,mi,mj->m", J, x, x)
        field = -(x * h).sum(axis=1)
        energy = inter + field
        return energy
