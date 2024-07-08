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

    mat = Mat(100, length=25, width=2., thickness=0.5, sizeX=50., sizeY=50., sizeZ=50., shear=1., tensile=2500.)
    net = Net(mat, periodic=True)
    stack = Stack(net, threshold=10)
    mesh = Mesh(stack)

    # Instantiate the model
    model = Model(mesh)
    # Permutation of indices
    perm = sp.sparse.csgraph.reverse_cuthill_mckee(model.P, symmetric_mode=True)
    # Visualize the system
    _, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    plot_system(model.stiffness(), model.constraint(), perm=None, ax=ax[0])
    plot_system(model.stiffness(), model.constraint(), perm=perm, ax=ax[1])
    plt.show()

    sol = solve(
        Model(mesh),
        packing=4.,
        solve=lambda A, b: sp.sparse.linalg.spsolve(A, b, use_umfpack=False),
        perm=sp.sparse.csgraph.reverse_cuthill_mckee(Model(mesh).P, symmetric_mode=True),
    )

    # Visualize system evolution
    _, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    plot_system(sol.stiffness(0), sol.constraint(0), ax=ax[0])
    plot_system(sol.stiffness(1), sol.constraint(1), ax=ax[1])
    plt.show()

    msh = vtk_mesh(
        mesh,
        sol.displacement(1),
        sol.rotation(1),
        sol.force(1),
        sol.torque(1),
    )
    msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)
