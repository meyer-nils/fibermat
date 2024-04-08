#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

from fibermat import *
from fibermat import Mat, Mesh


################################################################################
# Degrees of Freedom
################################################################################

def displacement(u):
    """ Return nodal displacements."""
    return u[...]


def rotation(u):
    """ Return nodal rotations."""
    return np.zeros_like(u)


def force(F):
    """ Return nodal forces."""
    return F[...]


def torque(F):
    """ Return nodal torques."""
    return np.zeros_like(F)


################################################################################
# Mechanical model
################################################################################

def stiffness(mesh, **kwargs):
    """
    Assemble the quadratic system to be minimized.

    """
    # Optional
    if mesh is None:
        mesh = Mesh()

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
        i, i,
        j, j,
    ]).ravel()
    col = np.array([
        i, j,
        i, j,
    ]).ravel()
    data = np.array([
         k0, -k0,
        -k0,  k0,
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


def constraint(mesh, **kwargs):
    """
    Assemble the linear constraints.

    """
    # Optional
    if mesh is None:
        mesh = Mesh()

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
    Z = (mesh.z.values + 0.5 * mesh.h.values).max()  # : upper boundary position
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

    # Assemble the quadratic programming system
    K, u, F, du, dF = stiffness(mesh)
    C, f, H, df, dH = constraint(mesh)
    P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')
    # Permutation of indices
    perm = sp.sparse.csgraph.reverse_cuthill_mckee(P, symmetric_mode=True)
    # Enhanced solver
    spsolve = lambda A, b: sp.sparse.linalg.spsolve(A, b, use_umfpack=False)
    # # Visualize the system
    # fig, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    # plot_system((K, u, F, du, dF), (C, f, H, df, dH), perm=None, ax=ax[0])
    # plot_system((K, u, F, du, dF), (C, f, H, df, dH), perm=perm, ax=ax[1])
    # plt.show()

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solve(
        mesh,
        stiffness(mesh),
        constraint(mesh),
        packing=4,
        solve=spsolve,
        perm=perm,
    )

    # # Visualize the system
    # fig, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    # plot_system((K, u(0), F(0), du, dF), (C, f(0), H(0), df, dH), ax=ax[0])
    # plot_system((K, u(1), F(1), du, dF), (C, f(1), H(1), df, dH), ax=ax[1])
    # plt.show()

    # Export as VTK
    msh = vtk_mesh(
        mesh,
        displacement(u(1)),
        rotation(u(1)),
        force(f(1) @ C),
        torque(f(1) @ C),
    )
    msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)
    # msh.save("outputs/msh.vtk")
