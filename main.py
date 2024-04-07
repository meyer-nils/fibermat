#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                                        ██╖
████████╖  ████┐  ████╖       ██╖      ██╓╜
██╔═════╝  ██╔██ ██╔██║       ██║    ██████╖
█████─╖    ██║ ███╓╜██║██████╖██████╖██║ ██║
██╔═══╝    ██║ ╘══╝ ██║██║ ██║██╓─██║██╟───╜
██║    ██┐ ██║      ██║███ ██║██║ ██║│█████╖
╚═╝    └─┘ ╚═╝      ╚═╝╚══╧══╝╚═╝ ╚═╝╘═════╝
 █████┐       █████┐       ██┐
██╔══██┐     ██╓──██┐      └─┘       █╖████╖
 ██╖ └─█████ └███ └─┘      ██╖██████╖██╔══█║
██╔╝  ██╔══██   ███╖ ████╖ ██║██║ ██║██║  └╜
│██████╓╜   ██████╓╜ ╚═══╝ ██║██████║██║
╘══════╝    ╘═════╝        ╚═╝██╔═══╝╚═╝
      Rennes                  ██║
                              ╚═╝
@author: François Mahé
@mail: francois.mahe@ens-rennes.fr
(Univ Rennes, ENS Rennes, CNRS, IPR - UMR 6251, F-35000 Rennes, France)

@project: FiberMat
@version: v1.0

License:
--------
MIT License

Copyright (c) 2024 François Mahé

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Description:
------------
A mechanical solver to simulate fiber packing and perform statistical analysis.

References:
-----------
Mahé, F. (2023). Statistical mechanical framework for discontinuous composites:
  application to the modeling of flow in SMC compression molding (Doctoral
  dissertation, École centrale de Nantes).

"""


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(100, length=25, width=2, thickness=0.5, tensile=2500)
    # Build the fiber network
    net = Net(mat, periodic=True)
    # Stack fibers
    net = Stack(mat, net, threshold=10)
    # Create the fiber mesh
    mesh = Mesh(net)

    # Assemble the quadratic programming system
    K, u, F, du, dF = stiffness(mat, mesh)
    C, f, H, df, dH = constraint(mat, mesh)
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
        stiffness(mat, mesh),
        constraint(mat, mesh),
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
        mat, mesh,
        displacement(u(1)), rotation(u(1)),
        force(f(1) @ C), torque(f(1) @ C)
    )
    msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)
    msh.save("outputs/msh.vtk")
