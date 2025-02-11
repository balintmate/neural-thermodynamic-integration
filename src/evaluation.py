import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import wandb
from distance_on_torus import dist2_on_torus
from functools import partial


def g_hist_one(x, bins):
    num_dims = x.shape[-1]
    dR = dist2_on_torus(x) ** 0.5
    mask = 1 - jnp.eye(len(x))
    dR = dR * mask - (1 - mask)
    R_hist = jnp.histogram(dR.reshape(-1), bins=bins)[0]
    g_hist = R_hist / (bins[1:] ** (num_dims - 1))
    return g_hist


@partial(jax.jit, static_argnames=["ddpm"])
def eval_one_batch(ddpm, params, bins_g, x_train, key):
    x_samples, logZ = ddpm.sample(
        params, key=key, num_samples=len(x_train), n=x_train.shape[1]
    )

    g_hist = jax.vmap(g_hist_one, in_axes=(0, None))
    g_train = g_hist(x_train, bins_g).sum(0)
    g_samples = g_hist(x_samples, bins_g).sum(0)

    return g_train, g_samples, logZ


def eval_model(dataloader, ddpm, target, num_batches):
    logdict = {}
    g_train, g_samples = 0, 0
    logZ = 0

    bins_g = jnp.linspace(0, 4 * target.sigma, 300)
    key = jax.random.PRNGKey(5)

    for i in range(num_batches):
        key = jax.random.split(key)[0]
        x_train, _ = dataloader.next()
        g1, g2, logZ_ = eval_one_batch(ddpm, ddpm.params, bins_g, x_train, key)
        logZ += logZ_
        g_train += g1
        g_samples += g2
    logZ /= num_batches
    n = dataloader.N
    fig = plt.figure(figsize=(5, 5))
    plt.plot(bins_g[1:], g_train, label="MC data", linewidth=3)
    plt.plot(bins_g[1:], g_samples, label="diffusion samples", linewidth=1.5)
    plt.legend()
    plt.yticks([])
    plt.ylim(bottom=0)
    plt.xlim(0, 3 * target.sigma)
    plt.xlabel("r$r/\sigma$", fontsize=18)
    plt.xticks(jnp.arange(4) * target.sigma, jnp.arange(4))
    plt.ylabel(r"$g(r)$", fontsize=18)
    logdict = {
        **logdict,
        f"g(r)/{n}": wandb.Image(fig),
        f"logZ/{n}": logZ,
    }
    plt.close()

    wandb.log(logdict)
