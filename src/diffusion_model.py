import jax
import jax.numpy as jnp


from jax.scipy.stats.norm import pdf as normpdf
from functools import partial
from src.distance_on_torus import dist2_on_torus


class diffusion_model:
    def __init__(
        self,
        num_integration_steps,
        sigma_min,
        sigma_max,
        target_system,
    ):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.num_integration_steps = num_integration_steps
        self.target_system = target_system

    def init_params(self, key, maxN):
        self.params = self.E_model.init(
            rngs=key,
            t=jnp.ones(
                1,
            ),
            x=jax.random.uniform(jax.random.PRNGKey(6), (maxN, self.num_features)),
            n=jnp.array([maxN]),
        )

    ## dx = f dt +  g dW
    ## f = 0; g**2 = beta
    ## sigma = sqrt(int_beta**2)
    def sigma(self, t):
        return (self.sigma_min ** (1 - t)) * (self.sigma_max**t)

    def beta(self, t):  # d/dt sigma^2(t) = 2 sigma * sigma'
        # min * (max/min)**t
        return 2 * self.sigma(t) ** 2 * jnp.log(self.sigma_max / self.sigma_min)

    def energy(self, params, x, t, n):
        E_NN = self.E_model.apply(params, x=x, t=t, n=n)
        # return E_NN * sigma_min
        ## fixing boundary coniditons
        R2 = dist2_on_torus(x)
        mask = jnp.array(jnp.arange(len(x)) < n, dtype="int32")
        mask = mask.reshape(-1, 1)
        mask = jnp.einsum("ie,jt->ij", mask, mask) * (1 - jnp.eye(len(x)))
        R2 = R2 + (1 - mask)  ## avoid small distances
        E_softLJ = (self.target_system.U_ij_soft(t[0], R2) * mask).sum()

        NN_w = (1 - t[0]) * t[0]
        LJ_w = (1 - t[0]) * self.sigma_min

        return NN_w * E_NN + LJ_w * E_softLJ

    def force(self, params, x, t, n):
        return -jax.grad(lambda x: self.energy(params, x=x, t=t, n=n))(x)

    def loss_fn(self, params, batch, key):
        x0, n = batch
        key1, key2 = jax.random.split(key, 2)

        t = jax.random.uniform(key1, (len(x0), 1))

        z = jax.random.normal(key2, x0.shape)

        def loss_one(x0, n, z, t):
            xt = x0 + self.sigma(t[0]) * z  #
            xt = xt % 1
            mask = (jnp.arange(len(x0)) < n[0]).reshape(-1, 1)
            ## score = force = - grad (logp) -z/sigma
            force_times_sigma = self.force(params, xt, t, n)
            z_pred = -force_times_sigma
            error2 = (z_pred - z) ** 2
            return (error2 * mask).mean()

        return jax.vmap(loss_one)(x0, n, z, t).mean()

    @partial(jax.jit, static_argnames=["self", "num_samples", "n"])
    def sample(self, params, key, num_samples, n):
        x = jax.random.uniform(key, (num_samples, n, self.num_features))
        n = jnp.array([n])
        logZ = jnp.array(0.0)
        dt = 1 / self.num_integration_steps
        t = jnp.array([1.0])
        init_value = (x, logZ, t, key)

        def body_fun(i, carry):
            x, logZ, t, key = carry
            z = jax.random.normal(key, x.shape)

            def rescaled_E(x, t):  # absorb the division by sigma
                return self.energy(params, x=x, t=t, n=n) / self.sigma(t)[0]

            def force_dlogZ(x, t):
                dUdx, dUdt = jax.grad(rescaled_E, argnums=(0, 1))(x, t)
                return -dUdx, dUdt

            score, dlogZ = jax.vmap(force_dlogZ, in_axes=(0, None))(x, t)

            drift = self.beta(t[0]) * score
            x += drift * dt + z * (self.beta(t[0]) * dt) ** 0.5
            return x % 1, logZ + dt * dlogZ.mean(), t - dt, jax.random.split(key)[0]

        x, logZ, t, key = jax.lax.fori_loop(
            0, self.num_integration_steps, body_fun, init_value
        )

        return x, logZ
