#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FiberMat
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

License
-------
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

Description
-----------
A mechanical solver to simulate fiber packing and perform statistical analysis.

References
----------
Mahé, F. (2023). Statistical mechanical framework for discontinuous composites:
  application to the modeling of flow in SMC compression molding (Doctoral
  dissertation, École centrale de Nantes).

Structure
---------
+ mat.py
    └── Mat
+ net.py
    ├── Net
    └── Stack
+ mesh.py
    └── Mesh
+ solver.py
    ├── solve
    └── plot_system
+ model
    └── timoshenko.py
        ├── displacement
        ├── rotation
        ├── force
        ├── torque
        ├── stiffness
        └── constraint
+ utils
    └── interpolation.py
        └──  Interpolate
    └── render.py
        ├── vtk_fiber
        ├── vtk_mat
        └── vtk_mesh

Example
-------
from fibermat import *

mat = Mat(100)
net = Net(mat)
stack = Stack(net)
mesh = Mesh(stack)

K, u, F, du, dF = stiffness(mesh)
C, f, H, df, dH = constraint(mesh)
P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')

K, C, u, f, F, H, Z, rlambda, mask, err = solve(
    mesh,
    stiffness(mesh),
    constraint(mesh),
    packing=4,
    solve=lambda A, b: sp.sparse.linalg.spsolve(A, b, use_umfpack=True),
    perm=sp.sparse.csgraph.reverse_cuthill_mckee(P, symmetric_mode=True),
)

msh = vtk_mesh(
    mesh,
    displacement(u(1)),
    rotation(u(1)),
    force(f(1) @ C),
    torque(f(1) @ C),
)
msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)

"""

import fibermat
from fibermat.mat import *
from fibermat.net import *
from fibermat.mesh import *
from fibermat.solver import *
from fibermat.model.timoshenko import *
from fibermat.utils import *

__author__ = "François Mahé"
__authors__ = ["François Mahé"]
__contact__ = "francois.mahe@ens-rennes.fr"
__copyright__ = "Copyright (c) 2024 François Mahé"
__credits__ = ["François Mahé"]
__date__ = "19/03/2024"
__deprecated__ = False
__email__ = "francois.mahe@ens-rennes.fr"
__header__ = """
████████╖██┐██╖                   ████┐  ████╖       ██╖
██╔═════╝└─┘██║    ██████╖█╖████╖ ██╔██ ██╔██║       ██║
█████─╖  ██╖██████╖██║ ██║██╔══█║ ██║ ███╓╜██║██████╖█████╖
██╔═══╝  ██║██║ ██║██╟───╜██║  └╜ ██║ ╘══╝ ██║██║ ██║██╔══╝
██║      ██║█████╓╜│█████╖██║     ██║      ██║███ ██║█████╖
╚═╝      ╚═╝╚════╝ ╘═════╝╚═╝     ╚═╝      ╚═╝╚══╧══╝╚════╝
 █████┐       █████┐       ██┐
██╔══██┐     ██╓──██┐      └─┘       █╖████╖
 ██╖ └─█████ └███ └─┘      ██╖██████╖██╔══█║
██╔╝  ██╔══██   ███╖ ████╖ ██║██║ ██║██║  └╜
│██████╓╜   ██████╓╜ ╚═══╝ ██║██████║██║
╘══════╝    ╘═════╝        ╚═╝██╔═══╝╚═╝
      Rennes                  ██║
                              ╚═╝
"""
__home__ = "https://github.com/fmahe/fibermat"
__license__ = "MIT"
__maintainer__ = "François Mahé"
__status__ = "Production"
__version__ = "1.0"
