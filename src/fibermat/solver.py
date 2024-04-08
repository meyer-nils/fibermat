#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import warnings
from matplotlib import pyplot as plt
from tqdm import tqdm

from fibermat import *
from fibermat.model.timoshenko import stiffness, constraint
from fibermat.utils.interpolation import Interpolate


def solve(mesh, stiffness, constraint, packing=1., itermax=1000,
          solve=sp.sparse.linalg.spsolve, perm=None, tol=1e-6,
          errtol=1e-6, interp_size=None, verbose=True, **kwargs):
    r"""
    An iterative mechanical solver for **fiber packing problems**.

    It solves the *quadratic programming problem*:

    .. MATH::
        \min_{\mathbf{u}, \mathbf{f}} \left(
            \frac{1}{2} \, \mathbf{u} \, \mathbb{K} \, \mathbf{u}
            - \mathbf{F} \, \mathbf{u}
            - \mathbf{f} \, (\mathbf{H} - \mathbb{C} \, \mathbf{u})
        \right)
    .. MATH::
        \quad s.t. \quad \mathbb{C} \, \mathbf{u} \leq \mathbf{H} \, ,
        \quad \mathbf{u} \leq 0 \, ,
        \quad \mathbf{f} \geq 0
        \quad and
        \quad \mathbf{f} \, (\mathbf{H} - \mathbb{C} \, \mathbf{u}) = 0

    where:
        - 𝐮 is the vector of generalized displacements (*unknowns of the problem*).
        - 𝐟 is the vector of generalized forces (*unknown Lagrange multipliers*).
        - 𝕂 is the stiffness matrix of the fiber set.
        - 𝐅 is the vector of external efforts.
        - ℂ is the matrix of non-penetration constraints.
        - 𝐇 is the vector of minimal distances between fibers (minimal distances).

    The *mechanical equilibrium* allows reformulating the problem as a system of inequalities:

    .. MATH::
        \Rightarrow \quad \left[\begin{matrix}
            \mathbb{K} & \mathbb{C}^T \\
            \mathbb{C} & 0
        \end{matrix}\right] \binom{\mathbf{u}}{\mathbf{f}}
            \leq \binom{\mathbf{F}}{\mathbf{H}}

    which is solved using an iterative *Updated Lagrangian Approach*.

    .. HINT::
        Models used to build the matrices are implemented in :ref:`🔧 Model`:
            - 𝕂 and 𝐅 : :func:`~.model.timoshenko.stiffness`.
            - ℂ and 𝐇 : :func:`~.model.timoshenko.constraint`.

    Parameters
    ----------
    mesh : pandas.DataFrame, optional
        Fiber mesh represented by a :class:`~.Mesh` object.
    stiffness : tuple
        K : sparse matrix
            Stiffness matrix (symmetric positive-semi definite).
        u : numpy.ndarray
            Displacement vector.
        F : numpy.ndarray
            Load vector.
        du : numpy.ndarray
            Incremental displacement vector.
        dF : numpy.ndarray
            Incremental load vector.
    constraint : tuple
        C : sparse matrix
            Constraint matrix.
        f : numpy.ndarray
            Force vector.
        H : numpy.ndarray
            Upper-bound vector.
        df : numpy.ndarray
            Incremental force vector.
        dH : numpy.ndarray
            Incremental upper-bound vector.
    packing : float, optional
        Targeted value of packing. Must be greater than 1. Default is 1.0.
    itermax : int, optional
        Maximum number of solver iterations. Default is 1000.

    Returns
    -------
    tuple
        K : sparse matrix
            Stiffness matrix (symmetric positive-semi definite).
        C : sparse matrix
            Constraint matrix.
        u : Interpolate
            Displacement vector.
        f : Interpolate
            Force vector.
        F : Interpolate
            Load vector.
        H : Interpolate
            Upper-bound vector.
        Z : Interpolate
            Upper-boundary position.
        rlambda : Interpolate
            Compaction stretch factor.
        mask : Interpolate
            Active rows and columns in the system of inequations.
        err : Interpolate
            Numerical error of the linear solver.

    .. SEEALSO::
        Simulation results are given as functions of a pseudo-time parameter (between 0 and 1) using :class:`~.interpolation.Interpolate` objects.

    Other Parameters
    ----------------
    solve : callable, optional
        Sparse solver. It is a callable object that takes as inputs a sparse
        symmetric matrix 𝔸 and a vector 𝐛 and returns the solution 𝐱 of the
        linear system: 𝔸 𝐱 = 𝐛. Default is `scipy.sparse.linalg.spsolve`.
    perm : numpy.ndarray, optional
        Permutation of indices.
    tol : float, optional
        Tolerance used for contact detection (mm). Default is 1e-6 mm.
    errtol : float, optional
        Tolerance for the numerical error. Default is 1e-6.
    interp_size : int, optional
        Size of array used for interpolation. Default is None.
    verbose : bool, optional
        If True, a progress bar is displayed. Default is True.
    kwargs :
        Additional keyword arguments ignored by the function.

    :Use:

        >>> # Generate a set of fibers
        >>> mat = Mat(100)
        >>> # Build the fiber network
        >>> net = Net(mat)
        >>> # Create the fiber mesh
        >>> mesh = Mesh(net)
        >>> # Solve the mechanical packing problem
        >>> K, C, u, f, F, H, Z, rlambda, mask, err = solve(mesh, stiffness(mesh), constraint(mesh), packing=4)

    """
    # Optional
    if mesh is None:
        mesh = Mesh()

    # Assemble the quadratic programming system
    K, u, F, du, dF = stiffness
    C, f, H, df, dH = constraint
    P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')
    x = np.r_[u, f]
    q = np.r_[F, H]
    dx = np.r_[du, df]
    dq = np.r_[dF, dH]

    if perm is None:
        perm = np.arange(P.shape[0])

    u, f = np.split(x, [K.shape[0]])  # Memory-shared
    F, H = np.split(q, [K.shape[0]])  # Memory-shared
    du, df = np.split(dx, [K.shape[0]])  # Memory-shared
    dF, dH = np.split(dq, [K.shape[0]])  # Memory-shared

    u_ = [u.copy()]
    f_ = [f.copy()]
    F_ = [F.copy()]
    H_ = [H.copy()]
    Z_ = [(mesh.z.values + 0.5 * mesh.h.values).max()]
    rlambda_ = [1.0]
    mask_ = [(np.real(q - P @ x) <= tol)]
    err_ = [0]

    # Incremental solver
    with tqdm(total=itermax, desc="Packing: {:.2}".format(rlambda_[-1]),
              disable=not verbose) as pbar:
        i = 0
        while (i < pbar.total) and (rlambda_[-1] < packing):
            # Solve step
            dx *= 0
            # mask = [True] * K.shape[0] + [*(C @ u >= H - tol)]
            mask = (np.real(q - P @ x) <= tol)
            mask &= np.array(np.sum(np.abs(np.real(P)), axis=0) > 0).ravel()
            # Solve linear problem
            # dx[mask] += np.real(solve(P[np.ix_(mask, mask)], dq[mask]))
            dx[perm[mask[perm]]] = np.real(
                solve(
                    P[np.ix_(perm[mask[perm]], perm[mask[perm]])],
                    dq[perm[mask[perm]]]
                )
            )
            # Calculate error
            err = np.linalg.norm(P[np.ix_(mask, mask)] @ dx[mask] - dq[mask])
            # Calculate evolution
            d = np.real(q - P @ x)
            v = np.real(dq - P @ dx)

            try:
                # Calculate the next step
                dU = -min(d[(d > tol) & (v > 0)] / v[(d > tol) & (v > 0)])
                # Stopping criteria
                stop = False
                if err > errtol:
                    if verbose:
                        warnings.warn("Stopping criteria: err = {}".format(err),
                                      UserWarning)
                    stop = True
                if stop:
                    raise ValueError
            except ValueError:
                break

            # Jump to the next step
            x += dx * dU
            q += dq * dU
            # Store results
            u_ += [u.copy()]
            f_ += [f.copy()]
            F_ += [F.copy()]
            H_ += [H.copy()]
            Z_ += [Z_[-1] + dU]
            rlambda_ += [Z_[0] / Z_[-1]]
            mask_ += [mask.copy()]
            err_ += [err.copy()]

            # Update
            i += 1
            pbar.set_description("Packing: {:.2}".format(rlambda_[-1]))
            pbar.update()

    # Interpolate results
    with warnings.catch_warnings():
        # Ignore warning messages due to infinite values in 𝐇
        warnings.filterwarnings('ignore')
        u = Interpolate(u_, size=interp_size)
        f = Interpolate(f_, size=interp_size)
        F = Interpolate(F_, size=interp_size)
        H = Interpolate(H_, size=interp_size)
        Z = Interpolate(Z_, size=interp_size)
        rlambda = Interpolate(rlambda_)
        mask = Interpolate(mask_, kind='previous')
        err = Interpolate(err_, kind='previous')

    # Return interpolated results
    return K, C, u, f, F, H, Z, rlambda, mask, err


def plot_system(stiffness, constraint,
                solve=sp.sparse.linalg.spsolve, perm=None, tol=1e-6, ax=None):
    """
    Visualize the system of equations and calculate the step error.

    Parameters
    ----------
    stiffness : tuple
        K : sparse matrix
            Stiffness matrix (symmetric positive-semi definite).
        u : numpy.ndarray
            Displacement vector.
        F : numpy.ndarray
            Load vector.
        du : numpy.ndarray
            Incremental displacement vector.
        dF : numpy.ndarray
            Incremental load vector.
    constraint : tuple
        C : sparse matrix
            Constraint matrix.
        f : numpy.ndarray
            Force vector.
        H : numpy.ndarray
            Upper-bound vector.
        df : numpy.ndarray
            Incremental force vector.
        dH : numpy.ndarray
            Incremental upper-bound vector.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axes.

    Other Parameters
    ----------------
    solve : callable, optional
        Sparse solver. It is a callable object that takes as inputs a sparse
        symmetric matrix 𝔸 and a vector 𝒃 and returns the solution 𝒙 of the
        linear system: 𝔸 𝒙 = 𝒃. Default is `scipy.sparse.linalg.spsolve`.
    perm : numpy.ndarray, optional
        Permutation of indices.
    tol : float, optional
        Tolerance used for contact detection (mm). Default is 1e-6 mm.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes.

    """
    # Assemble the quadratic programming system
    K, u, F, du, dF = stiffness
    C, f, H, df, dH = constraint
    P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')
    x = np.r_[u, f]
    q = np.r_[F, H]
    dx = np.r_[du, df]
    dq = np.r_[dF, dH]

    mask0 = np.array([True] * K.shape[0] + [False] * C.shape[0])
    D0 = sp.sparse.diags(mask0.astype(float))
    mask1 = np.array([False] * K.shape[0] + [*(np.isclose(C @ u, H) & np.tile([True, False, False], C.shape[0] // 3))])
    D1 = sp.sparse.diags(mask1.astype(float))
    mask2 = np.array([False] * K.shape[0] + [*(np.isclose(C @ u, H) & np.tile([False, True, False], C.shape[0] // 3))])
    D2 = sp.sparse.diags(mask2.astype(float))
    mask3 = np.array([False] * K.shape[0] + [*(np.isclose(C @ u, H) & np.tile([False, False, True], C.shape[0] // 3))])
    D3 = sp.sparse.diags(mask3.astype(float))
    mask4 = np.array([False] * K.shape[0] + [*(~np.isclose(C @ u, H))])
    D4 = sp.sparse.diags(mask4.astype(float))

    if perm is None:
        perm = np.arange(P.shape[0])

    # Figure
    if ax is None:
        fig, ax = plt.subplots()
    ax.spy((D0 @ P @ D0)[np.ix_(perm, perm)], ms=3, color='black', alpha=0.5, label="stiffness")
    ax.spy((D2 @ P + P @ D2)[np.ix_(perm, perm)], ms=3, color='tab:blue', alpha=0.25, label="inner")
    ax.spy((D1 @ P + P @ D1)[np.ix_(perm, perm)], ms=3, color='tab:green', alpha=0.5, label="lower")
    ax.spy((D3 @ P + P @ D3)[np.ix_(perm, perm)], ms=3, color='tab:red', alpha=0.5, label="upper")
    ax.spy((D4 @ P + P @ D4)[np.ix_(perm, perm)], ms=1, color='gray', zorder=-1, alpha=0.1,
           label="inactive")
    ax.legend()

    mask = (np.real(q - P @ x) <= tol)
    mask &= np.array(np.sum(np.abs(np.real(P)), axis=0) > 0).ravel()

    # Solve linear problem
    dx[perm[mask[perm]]] = np.real(
        solve(
            P[np.ix_(perm[mask[perm]], perm[mask[perm]])],
            dq[perm[mask[perm]]]
        )
    )

    # Calculate error
    err = np.linalg.norm(P[np.ix_(mask, mask)] @ dx[mask] - dq[mask])
    print(err)

    return ax


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    # from fibermat import *

    # Generate a set of fibers
    mat = Mat(10)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solve(
        mesh,
        stiffness(mesh),
        constraint(mesh),
        packing=4,
    )

    # Deform the mesh
    mesh.z += displacement(u(1))

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    if len(mesh):
        # Draw elements
        for i, j, k in tqdm(zip(mesh.index, mesh.beam, mesh.constraint),
                            total=len(mesh), desc="Draw mesh"):
            # Get element data
            a, b, c = mesh.iloc[[i, j, k]][[*"xyz"]].values
            if mesh.iloc[i].s < mesh.iloc[j].s:
                # Draw intra-fiber connection
                plt.plot(*np.c_[a, b],
                         c=plt.cm.tab10(mesh.fiber.iloc[i] % 10))
            if mesh.iloc[i].z < mesh.iloc[k].z:
                # Draw inter-fiber connection
                plt.plot(*np.c_[a, c], '--ok',
                         lw=1, mfc='none', ms=3, alpha=0.2)
            if mesh.iloc[i].fiber == mesh.iloc[k].fiber:
                # Draw fiber end nodes
                plt.plot(*np.c_[a, c], '+k', ms=3, alpha=0.2)
    # Set drawing box dimensions
    ax.set_xlim(-0.5 * mesh.attrs["size"], 0.5 * mesh.attrs["size"])
    ax.set_ylim(-0.5 * mesh.attrs["size"], 0.5 * mesh.attrs["size"])
    plt.show()
