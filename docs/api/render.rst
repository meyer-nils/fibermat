🌐 Render
=========

vtk_fiber
~~~~~~~~~

.. image:: ../../images/vtk_fiber.png
    :width: 640

.. autofunction:: fibermat.utils.render.vtk_fiber

vtk_mat
~~~~~~~

.. image:: ../../images/vtk_mat.png
    :width: 640

.. autofunction:: fibermat.utils.render.vtk_mat

vtk_mesh
~~~~~~~~

.. image:: ../../images/vtk_mesh.png
    :width: 640

.. autofunction:: fibermat.utils.render.vtk_mesh

Example
~~~~~~~

.. code-block:: python

    from fibermat import *

    # Create a VTK fiber
    vtk_fiber().plot()

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat, periodic=True)
    # Stack fibers
    stack = Stack(net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Create a VTK mat
    vtk_mat(mat).plot()

    # Create a VTK mesh
    vtk_mesh(mesh).plot()

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solve(
        mesh,
        stiffness(mesh),
        constraint(mesh),
        packing=4,
    )

    # Export as VTK
    msh = vtk_mesh(
        mesh,
        displacement(u(1)), rotation(u(1)),
        force(f(1) @ C), torque(f(1) @ C)
    )
    msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)

.. image:: ../../images/vtk_force.png
    :width: 640
