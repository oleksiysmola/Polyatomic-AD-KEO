# For comparison we consider
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from vibrojet.eckart import eckart
from vibrojet.keo import Gmat
from vibrojet.taylor import deriv_list

jax.config.update("jax_enable_x64", True)

pi = np.pi
toRadians = pi/180

# Masses of O, H, H atoms
masses = [12.00000000, 15.99491463, 1.00782503223, 1.00782503223, 1.00782503223, 1.00782503223]

# Equilibrium values of valence coordinates
r1, r2, r3, r4, r5, alpha1, alpha2, alpha3, alpha4, Sa, Sb, tau = 1.42167426, 0.95631445, 1.09061209, 1.09061209, 1.09061209, 108.10210467*toRadians, 110.28974516*toRadians, 110.28974516*toRadians, 110.28974516*toRadians, 0.0, 0.0, pi/3
q0 = [r1, r2, r3, r4, r5, alpha1, alpha2, alpha3, alpha4, Sa, Sb, tau]

# Valence-to-Cartesian coordinate transformation
#   input: array of three valence coordinates
#   output: array of shape (number of atoms, 3)
#           containing Cartesian coordinates of atoms


@eckart(q0, masses)
def valence_to_cartesian(q):
    r1, r2, r3, r4, r5, alpha1, alpha2, alpha3, alpha4, Sa, Sb, tau = q
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, r1],
            [r2*jnp.sin(alpha1), 0.0, r1 - r2*jnp.cos(alpha1)],
            [r3*jnp.sin(alpha2)*jnp.cos(tau - jnp.sqrt(2)*Sb/3), r3*jnp.sin(alpha2)*jnp.sin(tau - jnp.sqrt(2)*Sb/3), r3*jnp.cos(alpha2)],
            [r4*jnp.sin(alpha3)*jnp.cos(tau + 4*pi/3 + Sa/jnp.sqrt(6) + jnp.sqrt(2)*Sb/3), r4*jnp.sin(alpha3)*jnp.sin(tau + 4*pi/3 + Sa/jnp.sqrt(6) + jnp.sqrt(2)*Sb/3), r4*jnp.cos(alpha3)],
            [r5*jnp.sin(alpha4)*jnp.cos(tau + 2*pi/3 - Sa/jnp.sqrt(6) + jnp.sqrt(2)*Sb/3), r5*jnp.sin(alpha4)*jnp.sin(tau + 2*pi/3 - Sa/jnp.sqrt(6) + jnp.sqrt(2)*Sb/3), r5*jnp.cos(alpha4)]
        ]
    )


# Generate list of multi-indices specifying the integer exponents
# for each coordinate in the Taylor series expansion

max_order = 8  # max total expansion order
deriv_ind = [
    elem
    for elem in itertools.product(*[range(0, max_order + 1) for _ in range(len(q0))])
    if sum(elem) <= max_order
]
print("max expansion order:", max_order)
print("number of expansion terms:", len(deriv_ind))

# Function for computing kinetic G-matrix for given masses of atoms
# and internal coordinates
func = lambda x: Gmat(x, masses, valence_to_cartesian)

# Compute Taylor series expansion coefficients
Gmat_coefs = deriv_list(func, deriv_ind, q0, if_taylor=True)