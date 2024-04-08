🔧 Model
========

Models are stored in the directory `fibermat.model`.

Degrees of Freedom
~~~~~~~~~~~~~~~~~~

.. autofunction:: fibermat.model.timoshenko.displacement
.. autofunction:: fibermat.model.timoshenko.rotation
.. autofunction:: fibermat.model.timoshenko.force
.. autofunction:: fibermat.model.timoshenko.torque

Mechanical model
~~~~~~~~~~~~~~~~

stiffness
---------

.. autofunction:: fibermat.model.timoshenko.stiffness

constraint
----------

.. autofunction:: fibermat.model.timoshenko.constraint

Example
~~~~~~~

.. code-block:: python

    from fibermat import *

    # Linear model (Ψ² ≫ 1)
    mat = Mat(1, length=1, width=1, thickness=1, shear=1, tensile=np.inf)
    net = Net(mat)
    mesh = Mesh(net)
    print("Linear (Ψ² ≫ 1) =")
    print(4 / np.pi * stiffness(mesh, coupling=1)[0].todense())
    print()

    # Timoshenko model (Ψ² = 1)
    mat = Mat(1, length=1, width=1, thickness=1, shear=2, tensile=2)
    net = Net(mat)
    mesh = Mesh(net)
    print("Timoshenko (Ψ² = 1) = 1 / 2 *")
    print(4 / np.pi * stiffness(mesh, coupling=1)[0].todense())
    print()

    # Euler model (Ψ² ≪ 1)
    mat = Mat(1, length=1, width=1, thickness=1, shear=1e12, tensile=12)
    net = Net(mat)
    mesh = Mesh(net)
    print("Euler (Ψ² ≪ 1) = 1 / 12 *")
    print(4 / np.pi * stiffness(mesh, coupling=1)[0].todense())
    print()

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Assemble the quadratic programming system
    K, u, F, du, dF = stiffness(mesh)
    C, f, H, df, dH = constraint(mesh)
    P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')
    # Permutation of indices
    perm = sp.sparse.csgraph.reverse_cuthill_mckee(P, symmetric_mode=True)
    # Visualize the system
    fig, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8))
    plot_system((K, u, F, du, dF), (C, f, H, df, dH), perm=None, ax=ax[0])
    plot_system((K, u, F, du, dF), (C, f, H, df, dH), perm=perm, ax=ax[1])
    plt.show()

.. code-block::

    Linear (Ψ² ≫ 1) =
        [[ 1.   0.5 -1.   0.5]
         [ 0.5  inf -0.5 -inf]
         [-1.  -0.5  1.  -0.5]
         [ 0.5 -inf -0.5  inf]]

    Timoshenko (Ψ² = 1) = 1 / 2 *
        [[ 1.          0.5        -1.          0.5       ]
         [ 0.5         1.33333333 -0.5        -0.83333333]
         [-1.         -0.5         1.         -0.5       ]
         [ 0.5        -0.83333333 -0.5         1.33333333]]

    Euler (Ψ² ≪ 1) = 1 / 12 *
        [[ 12.   6. -12.   6.]
         [  6.   4.  -6.   2.]
         [-12.  -6.  12.  -6.]
         [  6.   2.  -6.   4.]]

    2.888979232532619e-13

.. image:: ../../images/system.png
    :width: 1280
