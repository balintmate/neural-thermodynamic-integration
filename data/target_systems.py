from dataclasses import dataclass
import jax.numpy as jnp
from typing import Sequence
import jax


## eps of LJ is actually 2*eps, because i'm not dividing by 2 when summing LJ(D^2)
@dataclass
class TargetSystemAbs:

    def U_ij(self, r2):
        raise NotImplementedError

    def U_x(self, x):
        raise NotImplementedError


@dataclass
class LJ(TargetSystemAbs):
    beta: float = 1

    def U_ij_soft(self, a, r2):
        sr6 = (self.sigma**2 / (a * self.sigma**2 + r2)) ** 3
        U_ij = 4 * self.eps * (sr6**2 - sr6)
        return U_ij

    def U_ij(self, r2):
        return self.U_ij_soft(0, r2)

    def U_x(self, x):
        return jnp.array(0.0)


@dataclass
class LJ3D(LJ):
    num_dim: int = 3
    sigma: float = 1 / 6
    eps: float = 0.4
    data_path: str = "../data/LJ3D"
    num_samples: int = 30 * 1000
    burn_in: int = 10 * 1000
