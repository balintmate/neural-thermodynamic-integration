import jax.numpy as jnp
import jax


def dR_on_torus(x):

    ## (num_dim, x,y,num_dim)
    def diff_fn(a, b):
        return a - b

    diff_fn = jax.vmap(jax.vmap(diff_fn, in_axes=(None, 0)), in_axes=(0, None))
    dR1 = diff_fn(x, x)
    dR2 = diff_fn(x, x + 1)
    dR3 = diff_fn(1 + x, x)

    ### get absolute min
    dR = jnp.where(jnp.abs(dR1) < jnp.abs(dR2), dR1, dR2)
    dR = jnp.where(jnp.abs(dR) < jnp.abs(dR3), dR, dR3)
    return dR


def dist2_on_torus(x):
    return (dR_on_torus(x) ** 2).sum(-1)
