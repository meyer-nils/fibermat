#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from copy import copy
from scipy.interpolate import interp1d

from fibermat import *
from fibermat import Mesh
from fibermat.model.timoshenko import Timoshenko
from fibermat.utils.interpolation import Interpolate


class Shear(Timoshenko):

    ############################################################################
    # Degrees of Freedom
    ############################################################################

    def displacement(self, t: np.ndarray = None):
        """ Return nodal displacements."""
        return self.u(t)[...]

    def rotation(self, t: np.ndarray = None):
        """ Return nodal rotations."""
        return np.zeros_like(self.u(t))

    def force(self, t: np.ndarray = None):
        """ Return nodal forces."""
        return (self.f(t) @ self.C)[...]

    def torque(self, t: np.ndarray = None):
        """ Return nodal torques."""
        return np.zeros_like(self.force(t))

    ############################################################################
    # Mechanical model
    ############################################################################

    def stiffness(self, t=None, **_):
        r"""
        Assemble the quadratic system to be minimized.

        The mechanical model is built using a linear shear model:

        .. MATH::
            \mathbb{K}_e = \frac{Gbh}{l_e} \cdot \frac{\pi / 4}{1 + \frac{G}{E} \left( \frac{l_e}{h} \right)^2}
                \left[\begin{matrix}
                    1  & -1  \\
                   -1  &  1  \\
                \end{matrix}\right]
                \ , \quad \mathbf{F}_e =
                \left(\begin{matrix}
                    0 \\
                    0 \\
                \end{matrix}\right)

        where:
            - 𝑙ₑ is the length of the beam element.
            - 𝐺 is the shear modulus.
            - 𝑏 and h are the width and thickness of the fiber.

        The generalized displacement vector :math:`\mathbf{u} = (\dots, u_i, \dots)`
        (with 𝑢ᵢ being the vertical displacement of the i-th node)
        satisfies *mechanical equilibrium*:

        .. MATH::
            \mathbb{K} \, \mathbf{u} = \mathbf{F}

        Parameters:
        -----------
        t : array-like, optional
            Interpolation parameter between 0 and 1.

        Returns:
        --------
        tuple
            K : sparse matrix
                Stiffness matrix (symmetric positive-semi definite).
            u : numpy.ndarray
                Generalized displacement vector.
            F : numpy.ndarray
                External force vector.
            du : numpy.ndarray
                Increment of displacement vector.
            dF : numpy.ndarray
                Increment of external force vector.

        """
        # Optional
        if t is not None:
            try:
                K, _, _, du, dF = self.stiffness()
                u = self.u(t)
                F = self.F(t)
                return K, u, F, du, dF
            except TypeError:
                pass

        mesh = self.mesh
        assert Mesh.check(mesh)

        # Get mesh data
        mask = (mesh.index.values < mesh.beam.values)
        fiber = mesh.fiber[mask].values
        i = mesh.index[mask].values
        j = mesh.beam[mask].values

        # Get material data
        mat = mesh.flags.mat
        fiber = mat.loc[fiber]
        l = mesh.s.loc[j].values - mesh.s.loc[i].values

        # Timoshenko number : Ψ² = E / G * (h / l) ^ 2
        k0 = np.pi / 4 * fiber[[*"Gbh"]].prod(axis=1).values / l
        i *= 1
        j *= 1

        # Create stiffness data
        row = np.array([
            i + 0, i + 0,
            j + 0, j + 0,
        ]).ravel()
        col = np.array([
            i + 0, j + 0,
            i + 0, j + 0,
        ]).ravel()
        data = np.array([
             k0, -k0,
            -k0, k0,
        ]).ravel()

        # Initialize 𝕂 matrix
        K = sp.sparse.coo_matrix((data, (row, col)),
                                 shape=(1 * len(mesh), 1 * len(mesh)))

        # Initialize 𝒖 and 𝑭 vectors
        u = np.zeros(K.shape[0])
        F = np.zeros(K.shape[0])
        du = np.zeros(K.shape[0])
        dF = np.zeros(K.shape[0])

        return K, u, F, du, dF

    def constraint(self, t=None, **_):
        r"""
        Assemble the linear constraints.

        The contact model is built using **normal non-penetration conditions** [1]_:

        .. MATH::
            \mathbb{C}_e =
                \left[\begin{array}{rrrr}
                     -1  &  0  \\
                      1  & -1  \\
                      1  &  0  \\
                \end{array}\right]
                \ , \quad \mathbf{H}_e =
                \left(\begin{matrix}
                    z_A - \frac{1}{2} \, h_A \\
                    z_B - z_A - \frac{1}{2} \, (h_A + h_B) \\
                    Z - z_B - \frac{1}{2} \, h_B \\
                \end{matrix}\right)

        where:
            - :math:`z_A` and :math:`z_B` are the vertical positions of nodes A and B.
            - :math:`h_A` and :math:`h_B` are the fiber thicknesses at nodes A and B.

        The vector 𝐟 is the vector of Lagrangian multipliers that corresponds to contact forces.
        It satisfies *KKT conditions*:

        .. MATH::
            \mathbb{C} \, \mathbf{u} \leq \mathbf{H} \, ,
            \quad \mathbf{f} \geq 0
            \quad and \quad \mathbf{f} \, (\mathbf{H} - \mathbb{C} \, \mathbf{u}) = 0

        .. RUBRIC:: Footnotes

        .. [1] `Karush–Kuhn–Tucker conditions, Wikipedia <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`_.

        Parameters:
        -----------
        t : array-like, optional
            Interpolation parameter between 0 and 1.

        Returns:
        --------
        tuple
            C : sparse matrix
                Constraint matrix.
            f : numpy.ndarray
                Internal force vector.
            H : numpy.ndarray
                Minimum distance vector.
            df : numpy.ndarray
                Increment of internal force vector.
            dH : numpy.ndarray
                Increment of distance vector.

        """
        # Optional
        if t is not None:
            try:
                C, _, _, df, dH = self.constraint()
                f = self.f(t)
                H = self.H(t)
                return C, f, H, df, dH
            except TypeError:
                pass

        mesh = self.mesh
        assert Mesh.check(mesh)

        # Get mesh data
        mask = (mesh.index.values <= mesh.constraint.values)
        i = mesh.index[mask].values
        j = mesh.constraint[mask].values
        k = np.arange(len(i))
        O = i * 0  # : zero
        I = O + 1  # : one

        # Get material data
        mat = mesh.flags.mat
        mesh["h"] = mat.h.loc[mesh.fiber].values
        zi = mesh.z.loc[i].values
        zj = mesh.z.loc[j].values
        hi = mesh.h.loc[i].values
        hj = mesh.h.loc[j].values
        Z = np.max(mesh.z.values + 0.5 * mesh.h.values)  # : upper boundary position
        i *= 1
        j *= 1
        k *= 3

        # Create constraint data
        row = np.array([k, k + 1, k + 1, k + 2]).ravel()
        col = np.array([i, i, j, j]).ravel()
        data = np.array([-I, I, -I, I]).ravel()

        # Initialize ℂ matrix
        C = sp.sparse.coo_matrix((data, (row, col)),
                                 shape=(3 * len(mesh[mask]), 1 * len(mesh)))

        # Initialize 𝒇 and 𝑯 vectors
        f = np.zeros(C.shape[0])
        H = np.zeros(C.shape[0])
        df = np.zeros(C.shape[0])
        dH = np.zeros(C.shape[0])
        # (X₁ + u₁) ≥ ½h₁ ⟺ -u₁ ≤ X₁ - ½h₁
        H[::3] += zi - 0.5 * hi
        # (X₂ + u₂) - (X₁ + u₁) ≥ ½(h₁ + h₂) ⟺ u₁ - u₂ ≤ X₂ - X₁ - ½(h₁ + h₂)
        H[1::3] += zj - zi - 0.5 * (hi + hj)
        # (X₂ + u₂) ≤ Z - ½h₂ ⟺ u₂ ≤ Z - X₂ - ½h₂
        H[2::3] += Z - zj - 0.5 * hj
        dH[2::3] = 1
        # For end nodes
        H[1::3][mesh[mask].index == mesh[mask].constraint.values] = np.inf

        return C, f, H, df, dH


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    # from fibermat import *

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Instantiate the model
    model = Shear(mesh)
    # Permutation of indices
    perm = sp.sparse.csgraph.reverse_cuthill_mckee(model.P, symmetric_mode=True)
    # Visualize the system
    fig, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    plot_system(model.stiffness(), model.constraint(), perm=None, ax=ax[0])
    plot_system(model.stiffness(), model.constraint(), perm=perm, ax=ax[1])
    plt.show()
