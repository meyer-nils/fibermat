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
    net = Net(mat)
    # Stack fibers
    stack = Stack(net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Solve the mechanical packing problem
    sol = solve(Timoshenko(mesh), packing=4)

    # Create a VTK mat
    vtk_mat(mat).plot()

    # Create a VTK mesh
    vtk_mesh(mesh).plot()

    # Export as VTK
    msh = vtk_mesh(
        mesh,
        sol.displacement(1),
        sol.rotation(1),
        sol.force(1),
        sol.torque(1),
    )
    msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)

.. image:: ../../images/vtk_force.png
    :width: 640
