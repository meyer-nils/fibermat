import numpy as np
import pandas as pd

from fibermat import *


################################################################################
# Tests
################################################################################

def test_solver():
    """
    Test the solver for a problem of 3 fibers.

    """
    mat = Mat(3)
    mat.x = -5, 0, 5
    mat.y = 0, 0, 0
    mat.z = 0.5, 1.5, 2.5
    mat.u = 0, 1, 0
    mat.v = 1, 0, 1
    mat.G = 4 / np.pi * 10

    net = Net(mat)
    mesh = Mesh(net)

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solve(
        mesh,
        stiffness(mesh),
        constraint(mesh),
        packing=4,
    )

    assert np.allclose(
        f(1) @ C,
        np.array([
             0., 0.,  # Fiber 0
             0., 0.,  # Fiber 0
             0., 0.,  # Fiber 0
             0., 0.,  # Fiber 1
            -1., 0.,  # Fiber 1 (upper)
             1., 0.,  # Fiber 1 (lower)
             0., 0.,  # Fiber 1
             0., 0.,  # Fiber 2
             0., 0.,  # Fiber 2
             0., 0.,  # Fiber 2
        ])
    )


################################################################################
# Main
################################################################################

if __name__ == '__main__':

    test_solver()

    mat = Mat(3)
    mat.x = -5, 0, 5
    mat.y = 0, 0, 0
    mat.z = 0.5, 1.5, 2.5
    mat.u = 0, 1, 0
    mat.v = 1, 0, 1
    mat.G = 4 / np.pi * 10

    net = Net(mat)
    mesh = Mesh(net)

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solve(
        mesh,
        stiffness(mesh),
        constraint(mesh),
        packing=4,
    )

    # Deform the mesh
    mesh.z += u(1)[::2]

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
