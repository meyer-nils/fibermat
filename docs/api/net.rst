🕸️ Net(work)
============

Net
---

.. image:: ../../images/net.png
    :width: 640

.. autoclass:: fibermat.net.Net
    :members:

Example
~~~~~~~

.. code-block:: python

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat, periodic=False)

    # Get node data
    pairs = net[[*"AB"]].values  # size: (n x 2)
    abscissa = net[["sA", "sB"]].values.reshape(-1, 2, 1)  # size: (n x 2 x 1)
    points = (net[["xA", "yA", "zA", "xB", "yB", "zB"]]
              .values.reshape(-1, 2, 3))  # size: (n x 2 x 3)

    # Check data
    Net.check(net)  # or `net.check()`
    # -> returns True if correct, otherwise it raises an error.

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    if len(mat):
        # Draw fibers
        for i in tqdm(range(len(mat)), desc="Draw fibers"):
            # Get fiber data
            fiber = mat.iloc[i]
            # Calculate fiber end points
            A = fiber[[*"xyz"]].values - 0.5 * fiber.l * fiber[[*"uvw"]].values
            B = fiber[[*"xyz"]].values + 0.5 * fiber.l * fiber[[*"uvw"]].values
            plt.plot(*np.c_[A, B], c=plt.cm.tab10(i % 10))
    if len(points):
        # Draw contacts
        for point in tqdm(points, desc="Draw nodes"):
            plt.plot(*point.T, '--ok', lw=1, mfc='none', ms=3, alpha=0.2)
    # Set drawing box dimensions
    ax.set_xlim(-0.5 * net.attrs["size"], 0.5 * net.attrs["size"])
    ax.set_ylim(-0.5 * net.attrs["size"], 0.5 * net.attrs["size"])
    plt.show()


.. image:: ../../images/net.png
    :width: 640

--------------------------------------------------------------------------------

Stack
-----

.. image:: ../../images/stack.png
    :width: 640

.. autoclass:: fibermat.net.Stack
    :members:

Example
~~~~~~~

.. code-block:: python

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat, periodic=False)
    # Stack fibers
    stack = Stack(net)

    # Get the linear system
    C, mg, H, h = Stack.constraint(stack)
    linsol = Stack.solve(stack)
    # Contact force
    f = linsol.ineqlin.marginals
    # Resulting force
    load = 0.5 * f @ np.abs(C) + 0.5 * f @ C

    # Check data
    Stack.check(stack)  # or `stack.check()`
    # -> returns True if correct, otherwise it raises an error.

    # Normalize by fiber weight
    load /= np.pi / 4 * mat[[*"lbh"]].prod(axis=1).mean()
    # Get loaded nodes
    points = (stack[stack.A < stack.B][["xA", "yA", "zA", "xB", "yB", "zB"]]
              .values.reshape(-1, 2, 3))
    # Prepare color scale
    cmap = plt.cm.viridis
    color = interp1d([np.min(load), np.max(load)], [0, 1])

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    if len(mat):
        # Draw fibers
        for i in tqdm(range(len(mat)), desc="Draw fibers"):
            # Get fiber data
            fiber = mat.iloc[i]
            # Calculate fiber end points
            A = fiber[[*"xyz"]].values - 0.5 * fiber.l * fiber[[*"uvw"]].values
            B = fiber[[*"xyz"]].values + 0.5 * fiber.l * fiber[[*"uvw"]].values
            plt.plot(*np.c_[A, B], c=cmap(color(load[i])))
    if len(points):
        # Draw contacts
        for point in tqdm(points[~np.isclose(f, 0)], desc="Draw nodes"):
            plt.plot(*point.T, '--ok', lw=1, mfc='none', ms=3, alpha=0.2)
    # Set drawing box dimensions
    ax.set_xlim(-0.5 * stack.attrs["size"], 0.5 * stack.attrs["size"])
    ax.set_ylim(-0.5 * stack.attrs["size"], 0.5 * stack.attrs["size"])
    # Add a color bar
    norm = plt.Normalize(vmin=np.min(load), vmax=np.max(load))
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(smap, ax=ax)
    cbar.set_label("Load / $mg$ ($N\,/\,N$)")
    plt.show()

.. image:: ../../images/stack.png
    :width: 640
