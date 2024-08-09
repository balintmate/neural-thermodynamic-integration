import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from distance_on_torus import dist2_on_torus, dR_on_torus
from data.target_systems import TargetSystemAbs
from functools import partial


class E_model(nn.Module):
    target_system: TargetSystemAbs
    NN: Sequence[int] = (64, 64)
    cutoff: float = 2 / 6
    size_to_pad: int = 1800
    ## for 2d
    # cutoff: float = 3 / 10
    # size_to_pad: int = 4000
    # cutoff: float = 1
    # size_to_pad: int = 110**2
    num_features: int = 32
    num_vec_features: int = 8
    agg_norm: int = 10
    num_layers: int = 3

    @nn.compact
    def __call__(self, t, x, n):

        num_dim = x.shape[-1]
        mask_node = jnp.array(jnp.arange(len(x)) < n, dtype="int32")
        mask2 = jnp.einsum("i,j->ij", mask_node, mask_node)
        dR = dR_on_torus(x) / self.cutoff
        D2 = (dR**2).sum(-1) + (1 - mask2) * 10  # filling particles are far

        def sizetopad(size):
            if type(self.size_to_pad) == int:
                return self.size_to_pad
            else:
                keys = filter(lambda k: k >= size, list(self.size_to_pad.keys()))
                key = min(keys)
                return self.size_to_pad[key]

        edges = jnp.stack(
            jnp.where(
                (D2 < 1) * (D2 > 0),
                size=sizetopad(len(x)),
                fill_value=-42,
            )
        )
        senders, receivers = edges[0], edges[1]

        edge_dist2 = D2.reshape(-1)[senders * len(D2) + receivers]
        mask_edge = senders != -42
        edge_dR = dR.reshape(-1, num_dim)[senders * len(D2) + receivers]
        edge_dR = jnp.expand_dims(edge_dR, 1)

        ## particle type could be added here later
        h = jax.nn.one_hot(jnp.zeros((len(x),)), 2)
        h = jax.vmap(nn.Dense(self.num_features, use_bias=False))(h)
        ###

        h_vec = jnp.zeros((len(x), self.num_vec_features, num_dim))

        edge_embedder = nn.Dense(self.num_vec_features, use_bias=False)
        edge_embedder = jax.vmap(edge_embedder, in_axes=-1, out_axes=-1)
        h_edge_vec = jax.vmap(edge_embedder)(edge_dR)

        ## particle type should be addded here as well (src,target)
        h_edge = MLP(self.NN + (h.shape[1],))(edge_dist2.reshape(-1, 1))

        ######
        for _ in range(self.num_layers):
            dh, dh_vec, dh_edge, dh_edge_vec = Layer(self.NN, self.agg_norm)(
                t,
                h,
                h_vec,
                h_edge,
                h_edge_vec,
                edge_dist2,
                edge_dR,
                mask_edge,
                senders,
                receivers,
            )
            h += dh
            h_vec += dh_vec
            h_edge += dh_edge
            h_edge_vec += dh_edge_vec
        return jnp.einsum("nf,n->", h, mask_node) / self.agg_norm


class Layer(nn.Module):
    NN: Sequence[int]
    agg_norm: float

    @nn.compact
    def __call__(
        self,
        t,
        h,
        h_vec,
        h_edge,
        h_edge_vec,
        edge_dist2,
        edge_dR,
        mask_edge,
        senders,
        receivers,
    ):

        inp = jnp.concatenate(
            [
                jnp.einsum("nfx,nfx->nf", h_vec[receivers], h_edge_vec),
                jnp.einsum("nfx,nfx->nf", h_vec[senders], h_edge_vec),
                jnp.einsum("nfx,nfx->nf", h_vec[senders], h_vec[receivers]),
                jnp.einsum("nfx,nfx->nf", h_vec[senders], h_vec[senders]),
                jnp.einsum("nfx,nfx->nf", h_vec[receivers], h_vec[receivers]),
                jnp.einsum("nfx,nfx->nf", h_edge_vec, h_edge_vec),
                h[senders],
                h[receivers],
                h_edge,
            ],
            -1,
        )
        # print(
        #     h.shape,
        #     h_vec.shape,
        #     h_edge.shape,
        #     h_edge_vec.shape,
        #     edge_dist2.shape,
        #     inp.shape,
        # )

        ## Message passing

        message_w_model = MessageWeight(self.NN + (h.shape[1] + h_vec.shape[1],))
        message_w_model = partial(message_w_model, t)
        message_w_model = jax.vmap(message_w_model)

        mw, mw_vec = jnp.split(message_w_model(inp), [h.shape[1]], axis=-1)

        ## smooth cutoff
        cutoff = 0.5 * (jnp.cos(edge_dist2 * jnp.pi) + 1)
        mw = jnp.einsum("nf,n->nf", mw, cutoff)
        mw_vec = jnp.einsum("nf,n->nf", mw_vec, cutoff)

        m = jnp.einsum("efx,ef,e->efx", h_edge_vec, mw_vec, mask_edge)
        h_vec = jnp.zeros(h_vec.shape).at[receivers].add(m) / self.agg_norm

        # h_vec_scale = jax.vmap(MLP(self.NN + (h_vec.shape[1],)))(h)
        # h_vec = jnp.einsum("nfx,nf->nfx", h_vec, h_vec_scale)

        m = jnp.einsum("ef,ef,e->ef", mw, h[senders], mask_edge)
        h = jnp.zeros(h.shape).at[receivers].add(m) / self.agg_norm

        ########################
        ### per_atom update
        h = jax.vmap(MLP(self.NN + (h.shape[1],)))(h)
        hvec_update = nn.Dense(h_vec.shape[1], use_bias=False)
        hvec_update = jax.vmap(hvec_update, in_axes=-1, out_axes=-1)
        h_vec = jax.vmap(hvec_update)(h_vec)
        ## per edge update
        edge_update = nn.Dense(h_edge_vec.shape[1], use_bias=False)
        edge_update = jax.vmap(edge_update, in_axes=-1, out_axes=-1)
        h_edge_vec = jnp.concatenate((h_edge_vec, h_vec[senders], h_vec[receivers]), 1)
        h_edge_vec = jax.vmap(edge_update)(h_edge_vec)

        h_edge = jnp.concatenate((h_edge, h[senders], h[receivers]), 1)
        h_edge = MLP(self.NN + (h.shape[1],))(h_edge)

        return h, h_vec, h_edge, h_edge_vec


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.swish(x)
        return x


class MessageWeight(nn.Module):
    NN: Sequence[int]

    @nn.compact
    def __call__(self, t, x):
        x = jnp.concatenate((x.reshape(-1), t.reshape(-1)))
        return MLP(self.NN)(x)
