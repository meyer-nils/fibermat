#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from copy import copy
from scipy.interpolate import interp1d

from fibermat import *
from fibermat import Mesh
from fibermat.utils.interpolation import Interpolate


class Timoshenko:
    r"""
    Mechanical model based on Timoshenko beam theory.

    It builds the system of inequalities:

    .. MATH::
        \Rightarrow \quad \left[\begin{matrix}
            \mathbb{K} & \mathbb{C}^T \\
            \mathbb{C} & 0
        \end{matrix}\right] \binom{\mathbf{u}}{\mathbf{f}}
            \leq \binom{\mathbf{F}}{\mathbf{H}}

    where:
        - 𝕂 is the stiffness matrix of the fiber set.
        - ℂ is the matrix of non-penetration constraints.
        - 𝐮 is the vector of generalized displacements (*unknowns of the problem*).
        - 𝐟 is the vector of internal forces (*unknown Lagrange multipliers*).
        - 𝐅 is the vector of external forces.
        - 𝐇 is the vector of minimum distances between fibers.

    Parameters:
    -----------
    mesh : pandas.DataFrame, optional
        Fiber mesh represented by a :class:`~.Mesh` object.

    ----

    Attributes
    ----------
    `K` :
        Stiffness matrix (symmetric positive-semi definite).
    `C` :
        Constraint matrix.
    `P` :
        Linear system.
    `x` :
        Solution vector.
    `dx` :
        Unknown vector.
    `q` :
        Upper-bound vector.
    `dq` :
        Increment vector.
    :attr:`u` :
        Generalized displacement vector.
    :attr:`f` :
        Internal force vector.
    :attr:`F` :
        External force vector.
    :attr:`H` :
        Minimum distance vector.
    :attr:`displacement` :
        Nodal displacements.
    :attr:`rotation` :
        Nodal rotations.
    :attr:`force` :
        Nodal forces.
    :attr:`torque` :
        Nodal torques.

    Methods
    -------
    :meth:`__call__` :
        Update mesh and return the quadratic programming problem.
    :meth:`copy` :
        Return a copy of the model.
    :meth:`set` :
        Calculate interpolated solution.
    :meth:`stiffness` :
        Assemble the quadratic system to be minimized.
    :meth:`constraint` :
        Assemble the linear constraints.

    ----

    """

    def __init__(self, mesh=None, **kwargs):
        """
        Parameters
        ----------
        mesh : pandas.DataFrame, optional
            Fiber mesh represented by a :class:`~.Mesh` object.

        Other Parameters
        ----------------
        kwargs :
            Additional keyword arguments passed to matrix constructors.

        """
        assert Mesh.check(mesh)

        # Set attributes
        self.mesh = mesh
        self.x = None
        self.q = None

        if mesh is not None:
            # Assemble the quadratic programming system
            K, u, F, du, dF = self.stiffness(**kwargs)
            C, f, H, df, dH = self.constraint(**kwargs)
            P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')
            x = np.r_[u, f]
            q = np.r_[F, H]
            dx = np.r_[du, df]
            dq = np.r_[dF, dH]

            # Add attributes
            self.K = K  # : stiffness matrix
            self.C = C  # : constraint matrix
            self.P = P  # : linear system
            self.x = x  # : solution vector
            self.q = q  # : upper-bound vector
            self.dx = dx  # : unknown vector
            self.dq = dq  # : increment vector

    # ~~~ Private methods ~~~ #

    def _split(self, x: np.ndarray):
        """ Split the solution vector into DoFs and Lagrangian multipliers."""
        indices = (self.K.shape[0],)
        return np.split(x, indices)  # Memory-shared

    # ~~~ Public methods ~~~ #

    def __call__(self, mesh=None, **kwargs):
        """ Update mesh and return the quadratic programming problem."""
        if mesh is not None:
            self.__init__(mesh, **kwargs)
        # Return quadratic programming system
        return self.mesh, self.P, self.x, self.q, self.dx, self.dq

    def copy(self):
        """ Return a copy of the model."""
        return copy(self)

    def set(self, x_, q_, **kwargs):
        """ Calculate interpolated solution."""
        self.x = Interpolate(x_, **kwargs)
        self.q = Interpolate(q_, **kwargs)

    ############################################################################
    # Degrees of Freedom
    ############################################################################

    def u(self, t: np.ndarray = None):
        """ Return generalized displacement vector."""
        return self._split(self.x(t))[0]

    def f(self, t: np.ndarray = None):
        """ Return internal force vector."""
        return self._split(self.x(t))[1]

    def F(self, t: np.ndarray = None):
        """ Return external force vector."""
        return self._split(self.q(t))[0]

    def H(self, t: np.ndarray = None):
        """ Return minimum distance vector."""
        return self._split(self.q(t))[1]

    def displacement(self, t: np.ndarray = None):
        """ Return nodal displacements."""
        return self.u(t)[..., ::2]

    def rotation(self, t: np.ndarray = None):
        """ Return nodal rotations."""
        return self.u(t)[..., 1::2]

    def force(self, t: np.ndarray = None):
        """ Return nodal forces."""
        return (self.f(t) @ self.C)[..., ::2]

    def torque(self, t: np.ndarray = None):
        """ Return nodal torques."""
        return (self.f(t) @ self.C)[..., 1::2]

    ############################################################################
    # Mechanical model
    ############################################################################

    def stiffness(self, t=None, lmin=0.01, lmax=None, coupling=0.99, **_):
        r"""
        Assemble the quadratic system to be minimized.

        The mechanical model is built using **Timoshenko beam theory** [1]_:

        .. MATH::
            \mathbb{K}_e = \frac{Gbh}{l_e} \cdot \frac{\pi / 4}{1 + \frac{G}{E} \left( \frac{l_e}{h} \right)^2}
                \left[\begin{matrix}
                    1  &  l_e / 2  &  -1  &  l_e / 2  \\
                    l_e / 2  &  {l_e}^2 / 3 + \frac{E}{G} h^2  &  -l_e / 2  &  {l_e}^2 / 6 - \frac{E}{G} h^2  \\
                   -1  &  -l_e / 2  &  1  &  -l_e / 2  \\
                    l_e / 2  &  {l_e}^2 / 6 - \frac{E}{G} h^2  &  -l_e / 2  &  {l_e}^2 / 3 + \frac{E}{G} h^2  \\
                \end{matrix}\right]
                \ , \quad \mathbf{F}_e =
                \left(\begin{matrix}
                    0 \\
                    0 \\
                    0 \\
                    0 \\
                \end{matrix}\right)

        where:
            - 𝑙ₑ is the length of the beam element.
            - 𝐸 is the tensile modulus.
            - 𝐺 is the shear modulus.
            - 𝑏 and h are the width and thickness of the fiber.

        The generalized displacement vector :math:`\mathbf{u} = (\dots, u_i, \theta_i, \dots)`
        (with 𝑢ᵢ being the vertical displacement and θᵢ the rotation of the cross-section of the i-th node)
        satisfies *mechanical equilibrium*:

        .. MATH::
            \mathbb{K} \, \mathbf{u} = \mathbf{F}

        .. RUBRIC:: Footnotes

        .. [1] `Timoshenko–Ehrenfest beam theory, Wikipedia <https://en.wikipedia.org/wiki/Timoshenko%E2%80%93Ehrenfest_beam_theory>`_.

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

        Other Parameters:
        -----------------
        lmin : float, optional
            Lower bound used to rescale beam lengths (mm). Default is 0.01 mm.
        lmax : float, optional
            Upper bound used to rescale beam lengths (mm).
        coupling : float, optional
            Coupling numerical constant between 0 and 1. Default is 0.99.
        _ :
            Additional keyword arguments ignored by the function.

        :Use:
            >>> # Linear model (Ψ² ≫ 1)
            >>> mat = Mat(1, length=1, width=1, thickness=1, shear=1, tensile=np.inf)
            >>> net = Net(mat)
            >>> mesh = Mesh(net)
            >>> model = Timoshenko(mesh)
            >>> # print("Linear (Ψ² ≫ 1) =")
            >>> print(4 / np.pi * model.stiffness(coupling=1)[0].todense())
            [[ 1.   0.5 -1.   0.5]
             [ 0.5  inf -0.5 -inf]
             [-1.  -0.5  1.  -0.5]
             [ 0.5 -inf -0.5  inf]]

            >>> # Timoshenko model (Ψ² = 1)
            >>> mat = Mat(1, length=1, width=1, thickness=1, shear=2, tensile=2)
            >>> net = Net(mat)
            >>> mesh = Mesh(net)
            >>> model = Timoshenko(mesh)
            >>> # print("Timoshenko (Ψ² = 1) = 1 / 2 *")
            >>> print(4 / np.pi * model.stiffness(coupling=1)[0].todense())
            [[ 1.          0.5        -1.          0.5       ]
             [ 0.5         1.33333333 -0.5        -0.83333333]
             [-1.         -0.5         1.         -0.5       ]
             [ 0.5        -0.83333333 -0.5         1.33333333]]

            >>> # Euler model (Ψ² ≪ 1)
            >>> mat = Mat(1, length=1, width=1, thickness=1, shear=1e12, tensile=12)
            >>> net = Net(mat)
            >>> mesh = Mesh(net)
            >>> model = Timoshenko(mesh)
            >>> # print("Euler (Ψ² ≪ 1) = 1 / 12 *")
            >>> print(4 / np.pi * model.stiffness(coupling=1)[0].todense())
            [[ 12.   6. -12.   6.]
             [  6.   4.  -6.   2.]
             [-12.  -6.  12.  -6.]
             [  6.   2.  -6.   4.]]

        """
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
        if lmin is None:
            lmin = np.min(l)
        if lmax is None:
            lmax = np.max(l)
        l = interp1d([min(np.min(l), lmin), max(np.max(l), lmax)],
                     [lmin, lmax])(l)

        # Timoshenko number : Ψ² = E / G * (h / l) ^ 2
        k0 = np.pi / 4 * fiber[[*"Gbh"]].prod(axis=1).values / l
        k0 /= (1 + (fiber.G / fiber.E) * (l / fiber.h) ** 2)
        k1 = k0 * l / 2
        k1 *= coupling  # Numerical regularization
        k2 = k0 * l ** 2 / 3
        k2 += k0 * (fiber.E / fiber.G) * fiber.h ** 2
        k3 = k0 * l ** 2 / 2
        k4 = k2 - k3
        i *= 2
        j *= 2

        # Create stiffness data
        row = np.array([
            i + 0, i + 0, i + 0, i + 0,
            i + 1, i + 1, i + 1, i + 1,
            j + 0, j + 0, j + 0, j + 0,
            j + 1, j + 1, j + 1, j + 1,
        ]).ravel()
        col = np.array([
            i + 0, i + 1, j + 0, j + 1,
            i + 0, i + 1, j + 0, j + 1,
            i + 0, i + 1, j + 0, j + 1,
            i + 0, i + 1, j + 0, j + 1,
        ]).ravel()
        data = np.array([
            k0, k1, -k0, k1,
            k1, k2, -k1, -k4,
            -k0, -k1, k0, -k1,
            k1, -k4, -k1, k2
        ]).ravel()

        # Initialize 𝕂 matrix
        K = sp.sparse.coo_matrix((data, (row, col)),
                                 shape=(2 * len(mesh), 2 * len(mesh)))

        # Initialize 𝒖 and 𝑭 vectors
        try:
            u = self.u(t)
        except TypeError:
            u = np.zeros(K.shape[0])
        try:
            F = self.F(t)
        except TypeError:
            F = np.zeros(K.shape[0])
        du = np.zeros(K.shape[0])
        dF = np.zeros(K.shape[0])

        return K, u, F, du, dF

    def constraint(self, t=None, **_):
        r"""
        Assemble the linear constraints.

        The contact model is built using **normal non-penetration conditions** [2]_:

        .. MATH::
            \mathbb{C}_e =
                \left[\begin{array}{rrrr}
                     -1  &  0  &  0  &  0  \\
                      1  &  0  & -1  &  0  \\
                      0  &  0  &  1  &  0  \\
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

        .. [2] `Karush–Kuhn–Tucker conditions, Wikipedia <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`_.

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

        Other Parameters:
        -----------------
        _ :
            Additional keyword arguments ignored by the function.

        """
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
        i *= 2
        j *= 2
        k *= 3

        # Create constraint data
        row = np.array([k, k + 1, k + 1, k + 2]).ravel()
        col = np.array([i, i, j, j]).ravel()
        data = np.array([-I, I, -I, I]).ravel()

        # Initialize ℂ matrix
        C = sp.sparse.coo_matrix((data, (row, col)),
                                 shape=(3 * len(mesh[mask]), 2 * len(mesh)))

        # Initialize 𝒇 and 𝑯 vectors
        try:
            f = self.f(t)
        except TypeError:
            f = np.zeros(C.shape[0])
        try:
            H = self.H(t)
        except TypeError:
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
    model = Timoshenko(mesh)
    # Permutation of indices
    perm = sp.sparse.csgraph.reverse_cuthill_mckee(model.P, symmetric_mode=True)
    # Visualize the system
    fig, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    plot_system(model.stiffness(), model.constraint(), perm=None, ax=ax[0])
    plot_system(model.stiffness(), model.constraint(), perm=perm, ax=ax[1])
    plt.show()
