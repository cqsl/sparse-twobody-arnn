#!/usr/bin/env python3

import time
from functools import partial

import equinox as eqx
import jax
import numpy as np
import optax
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import Array
from scipy.sparse import load_npz

from args import args
from ham import GeneralSpinsModel, exact
from nets import ARNNDense, TwoBoOnlySkip

dtype = np.float32


def leaf_size_real_nonzero(x):
    # Filter out equinox static fields
    if not isinstance(x, Array):
        return 0

    # If some but not all elements are exactly float zero, that means they are masked
    size = (x != 0).sum()
    if size == 0:
        size = x.size

    if np.iscomplexobj(x):
        size *= 2

    return size


def tree_size_real_nonzero(tree):
    return sum(tree_leaves(tree_map(leaf_size_real_nonzero, tree)))


def load_ham(ham_path):
    J = load_npz(ham_path).toarray().astype(dtype)
    N = J.shape[0]
    assert J.shape == (N, N)
    J = np.triu(J + J.T, k=1)

    J = J[None, ...]
    h = np.zeros((1, N), dtype=dtype)
    ham = GeneralSpinsModel(batch_size=1, N=N, J=J, h=h, dtype=dtype)
    return ham


def run_exact(ham, betas):
    if ham.N > 24:
        raise ValueError(f"Warning: N = {ham.N} is too large for exact enumeration")

    for beta in betas:
        start_time = time.time()
        stats = exact(ham, beta)
        used_time = time.time() - start_time

        F = stats["free_energy"]
        E = stats["energy"]
        S = stats["entropy"]
        M_abs = stats["|M|"]
        print(
            f"beta: {beta:.3g}",
            f"F: {F:.8g}",
            f"E: {E:.8g}",
            f"S: {S:.8g}",
            f"|M|: {M_abs:.8g}",
            f"time: {used_time:.3f}",
        )


@partial(jax.jit, static_argnames="optimizer")
def update(net, opt_state, key, ham, optimizer, beta):
    key, key_sample = jax.random.split(key)
    N = ham.N
    ham_params = ham.J[0]
    x, x_hat = net.sample(args.batch_size, N, ham_params, beta, key_sample)

    log_q = net.get_log_p(x, x_hat)
    energy = ham(x)
    loss = log_q + beta * energy

    params, static = eqx.partition(net, eqx.is_array)

    def loss_fun(params):
        net = eqx.combine(params, static)
        return ((loss - loss.mean()) * net(x, ham_params, beta)).mean()

    grads = jax.grad(loss_fun)(params)
    updates, opt_state = optimizer.update(grads, opt_state, net)
    net = eqx.apply_updates(net, updates)

    E = energy / N
    S = -log_q / N
    F = E - S / beta

    F_mean = F.mean()
    F_std = F.std()
    E_mean = E.mean()
    E_min = E.min()
    S_mean = S.mean()

    M = x.mean(axis=1)
    M_abs_mean = abs(M).mean()

    return net, opt_state, key, F_mean, F_std, E_mean, E_min, S_mean, M_abs_mean


def train(net, opt_state, key, ham, optimizer, beta, n_steps):
    for step in range(n_steps):
        start_time = time.time()
        net, opt_state, key, F, F_std, E, E_min, S, M_abs = update(
            net, opt_state, key, ham, optimizer, beta
        )
        used_time = time.time() - start_time

        print(
            f"beta: {beta:.3g}",
            f"step: {step}",
            f"F: {F:.8g}",
            f"F_std: {F_std:.8g}",
            f"E: {E:.8g}",
            f"E_min: {E_min:.8g}",
            f"S: {S:.8g}",
            f"|M|: {M_abs:.8g}",
            f"time: {used_time:.3f}",
        )

    return net, opt_state, key


def run_vmc(ham, betas):
    key = jax.random.PRNGKey(args.opt_seed)
    key, key_net = jax.random.split(key)
    if args.net_type == "twobo":
        net = TwoBoOnlySkip(
            N=ham.N,
            J=ham.J[0],
            param_dtype=dtype,
            key=key_net,
            weight_skip=args.weight_skip,
            use_beta_skip=args.use_beta_skip,
        )
    elif args.net_type == "made":
        net = ARNNDense(
            N=ham.N,
            n_ham_params=0,
            layers=1,
            features=1,
            param_dtype=dtype,
            key=key_net,
        )
    else:
        raise ValueError(f"Unknown net_type: {args.net_type}")

    n_params = tree_size_real_nonzero(net)
    print("n_params:", n_params)

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    net, opt_state, key = train(
        net, opt_state, key, ham, optimizer, betas[0], args.warmup_steps
    )
    for beta in betas:
        net, opt_state, key = train(
            net, opt_state, key, ham, optimizer, beta, args.opt_steps
        )


def main():
    ham = load_ham(args.ham_path)

    beta_start, beta_stop, beta_num = args.beta_range.split(",")
    beta_start = float(beta_start)
    beta_stop = float(beta_stop)
    beta_num = int(beta_num)
    betas = np.linspace(beta_start, beta_stop, beta_num)

    if args.net_type == "exact":
        run_exact(ham, betas)
    else:
        run_vmc(ham, betas)


if __name__ == "__main__":
    main()
