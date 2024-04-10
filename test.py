from fibermat import *

mesh = Mesh(Stack(Net(Mat(100))))

sol = solve(Timoshenko(mesh), packing=4)

vtk_mesh(mesh, sol.displacement(1)).plot()
