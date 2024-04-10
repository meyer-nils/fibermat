🔧 Model
========

Models are stored in the directory `fibermat.model`.

.. autoclass:: fibermat.model.timoshenko.Timoshenko
    :members: set, copy, __call__

Degrees of Freedom
~~~~~~~~~~~~~~~~~~

.. automethod:: fibermat.model.timoshenko.Timoshenko.u
.. automethod:: fibermat.model.timoshenko.Timoshenko.f
.. automethod:: fibermat.model.timoshenko.Timoshenko.F
.. automethod:: fibermat.model.timoshenko.Timoshenko.H
.. automethod:: fibermat.model.timoshenko.Timoshenko.displacement
.. automethod:: fibermat.model.timoshenko.Timoshenko.rotation
.. automethod:: fibermat.model.timoshenko.Timoshenko.force
.. automethod:: fibermat.model.timoshenko.Timoshenko.torque

Mechanical model
~~~~~~~~~~~~~~~~

stiffness
---------

.. autofunction:: fibermat.model.timoshenko.Timoshenko.stiffness

constraint
----------

.. autofunction:: fibermat.model.timoshenko.Timoshenko.constraint

Example
~~~~~~~

.. code-block:: python

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(net)
    # Create the fiber mesh
    mesh = Mesh(stack)
    # Instantiate the model
    model = Timoshenko(mesh)

    # Permutation of indices
    perm = sp.sparse.csgraph.reverse_cuthill_mckee(model.P, symmetric_mode=True)
    # Visualize the system
    fig, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    plot_system(model.stiffness(), model.constraint(), perm=None, ax=ax[0])
    plot_system(model.stiffness(), model.constraint(), perm=perm, ax=ax[1])
    plt.show()

.. image:: ../../images/system.png
    :width: 1280
