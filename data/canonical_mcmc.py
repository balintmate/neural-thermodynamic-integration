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

        samples_list = [x0]

        i = 0
        NUM_TO_SAMPLE = self.target_system.num_samples + self.target_system.burn_in
        while len(samples_list) < NUM_TO_SAMPLE:
            i = i + 1
            key1, key2, key = jax.random.split(key, 3)

            proposal = self.propose(samples_list[-1], key1)
            U_diff = self.U(proposal) - self.U(samples_list[-1])

            acceptance_prob = jnp.exp(-self.target_system.beta * U_diff)

            print(f"time: {time.time()-start:.0f}s", end="    ")
            print(
                f"accepted samples: {len(samples_list)+1}/{ NUM_TO_SAMPLE} ({100*(len(samples_list)+1)/( NUM_TO_SAMPLE):.1f}%)",
                end="    ",
            )
            print(f"acc. rate: {len(samples_list)/i:.3f}", end="    ")
            print(f"dx: {self.dx:.2e}", end="\r")
            sys.stdout.flush()

            if jax.random.uniform(key2, (1,))[0] < acceptance_prob:
                samples_list.append(proposal)

        print("")
        samples = jnp.stack(samples_list[self.target_system.burn_in :])

        samples = jax.random.permutation(jax.random.key(0), samples, axis=0)
        if not return_samples:
            with open(data_path, "wb") as file:
                pickle.dump(samples, file)
            print(f"Generated data has size {os.path.getsize(data_path)/2**20:.1f} MB")
            if wandb.run is not None:
                wandb.log({f"acceptance rate/{self.N}": len(samples_list) / i})
            print(50 * "-")
        else:
            return samples

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True
