#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from fibermat import Mat, Mesh, Net, Stack, Timoshenko, solve, vtk_mesh

# Generate a set of fibers
mat = Mat(300, length=25.4, width=4.0, thickness=0.14, sizeX=120, sizeY=20, sizeZ=20)
# Build the fiber network
net = Net(mat, periodic=False)
# Stack fibers
stack = Stack(net, threshold=1)
# Create the fiber mesh
mesh = Mesh(stack)
# Instantiate the model
model = Timoshenko(mesh)

# Solve the mechanical packing problem
sol = solve(model, packing=5.0, itermax=10000)

# Export as VTK
msh = vtk_mesh(
    mesh,
    sol.displacement(1),
    sol.rotation(1),
    sol.force(1),
    sol.torque(1),
)
msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)
msh.save("rectangle.vtk")
