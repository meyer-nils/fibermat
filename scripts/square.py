#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from fibermat import *


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    # Generate a set of fibers
    mat = Mat(3000, length=4, width=0.2, thickness=0.05, size=40, tensile=1600)
    # Build the fiber network
    net = Net(mat, periodic=True)
    # Stack fibers
    stack = Stack(net, threshold=1)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solve(
        mesh,
        stiffness(mesh),
        constraint(mesh),
        packing=4,
        itermax=10000,
    )

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
