from fibermat import *

mesh = Mesh(Stack(Net(Mat(10))))

K, C, u, f, F, H, Z, rlambda, mask, err = solve(
    mesh,
    stiffness(mesh),
    constraint(mesh),
    packing=4
)

vtk_mesh(mesh, displacement(u(1))).plot()
