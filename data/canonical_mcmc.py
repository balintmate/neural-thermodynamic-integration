import jax
import jax.numpy as jnp
from dataclasses import dataclass
from data.target_systems import TargetSystemAbs
from typing import Callable
import os, sys, pickle
from functools import partial
import time
from distance_on_torus import dist2_on_torus
import wandb
import optax


@dataclass
class Canonical_Sampler:
    target_system: TargetSystemAbs
    N: int

    @partial(jax.jit, static_argnames=["self"])
    def interaction(self, x):
        r2 = dist2_on_torus(x)
        r2 = r2 + jnp.eye(len(x))
        return self.target_system.U_ij(r2)

    @partial(jax.jit, static_argnames=["self"])
    def U(self, x):
        potential = self.target_system.U_x(x).sum()
        mask = 1 - jnp.eye(len(x))
        interaction = (self.interaction(x) * mask).sum()
        return potential + interaction

    @partial(jax.jit, static_argnames=["self"])
    def propose(self, x, key):
        z = jax.random.normal(key, x.shape) * jnp.sqrt(2 * self.dx)
        x = x + z
        x = x % 1
        return x

    # Monte Carlo sampling
    def sample(self, key, dx, return_samples=False):
        self.dx = dx
        data_path = self.target_system.data_path + f"_N={self.N}"
        start = time.time()
        print(50 * "-")
        print(f"N = {self.N}")
        if os.path.isfile(data_path) and not return_samples:
            print("Data already generated")
            return

        ## positon MC sampling

        x0 = jax.random.uniform(key, shape=(self.N, self.target_system.num_dim))

        # grad descent to spread out the points
        def loss(x):
            D = dist2_on_torus(x)
            mask = 1 - jnp.eye(len(x))
            return ((1 / (D + 1e-4)) * mask).sum()

        optim = optax.adam(learning_rate=1e-4)
        opt_state = optim.init(x0)

        D_min = 0
        while D_min < 1:
            grad = jax.grad(loss)(x0)
            updates, opt_state = optim.update(grad, opt_state, x0)
            x0 = optax.apply_updates(x0, updates) % 1
            mask = 1 - jnp.eye(len(x0))
            D = dist2_on_torus(x0)
            D = D + (1 - mask)
            D = D / self.target_system.sigma**2
            D_min = (D + mask).min()
            print(f"D2_min: {D_min:.4f}             ", end="\r")
        print()

        x_curr = x0
        samples_list = []
        i = 0

        @jax.jit
        def body_fn(i, carry):
            x_traj, U_traj, acc_prob_traj, key = carry
            key1, key2 = jax.random.split(key)
            x_curr = jax.tree.map(lambda x: x[i], x_traj)
            U_curr = U_traj[i]
            x_prop = self.propose(x_curr, key1)
            U_prop = self.U(x_prop)
            U_diff = U_prop - U_curr
            acc_prob = jnp.exp(-U_diff)
            acc_prob_traj = acc_prob_traj.at[i].set(jnp.clip(acc_prob, max=1))
            take_new = jax.random.uniform(key2, (1,))[0] < acc_prob
            x_new = jax.tree_map(
                lambda n, o: take_new * n + (1 - take_new) * o, x_prop, x_curr
            )
            x_traj = jax.tree.map(lambda x, n: x.at[i + 1].set(n), x_traj, x_new)
            U_new = take_new * U_prop + (1 - take_new) * U_curr
            U_traj = U_traj.at[i + 1].set(U_new)
            key = jax.random.split(key)[0]
            return x_traj, U_traj, acc_prob_traj, key

        NUM_TO_SAMPLE = self.target_system.num_samples + self.target_system.burn_in
        while i < NUM_TO_SAMPLE:
            N = 1000
            #######
            x_traj = jnp.zeros((N,) + x_curr.shape)
            x_traj = x_traj.at[0].set(x_curr)
            U_traj = jnp.zeros((N,))
            U_traj = U_traj.at[0].set(self.U(x_curr))
            acc_prob_traj = jnp.zeros(N)
            carry = (x_traj, U_traj, acc_prob_traj, key)
            x_traj, U_traj, acc_prob_traj, key = jax.lax.fori_loop(
                0, N - 1, body_fn, carry
            )
            x_curr = x_traj[-1]

            i += N
            if i > self.target_system.burn_in:
                samples_list.append(x_traj)

        i = 0

        print("")
        samples = jnp.concatenate(samples_list)
        samples = jax.random.permutation(jax.random.key(0), samples, axis=0)
        if not return_samples:
            with open(data_path, "wb") as file:
                pickle.dump(samples, file)
            print(f"Generated data has size {os.path.getsize(data_path)/2**20:.1f} MB")
            print(f"and shape {samples.shape}")
            if wandb.run is not None:
                wandb.log({f"acceptance rate/{self.N}": acc_prob_traj.mean()})
            print(50 * "-")
        else:
            return samples

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True
