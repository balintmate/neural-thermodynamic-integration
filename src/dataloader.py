import jax.numpy as jnp
import jax


class DataLoader:
    def __init__(self, x, n=None, batch_size=128):
        num_batches = len(x) // batch_size
        x = jax.random.permutation(jax.random.key(0), x, axis=0)
        if n is None:
            self.N = x.shape[1]
            n = jnp.full((len(x), 1), self.N)
        assert len(x) == len(n)
        self.x_all = jnp.stack(jnp.split(x[: num_batches * batch_size], num_batches))
        n = jax.random.permutation(jax.random.key(0), n, axis=0)
        self.n_all = jnp.stack(jnp.split(n[: num_batches * batch_size], num_batches))
        self.i = -1

    def next(self):
        self.i += 1
        x_batch = self.x_all[self.i % len(self.x_all)]
        n_batch = self.n_all[self.i % len(self.x_all)]
        return x_batch, n_batch
