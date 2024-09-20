SDyPy-view project
==================

The ``Plotter3D`` is a wrapper around the ``pyvista.BackgroundPlotter`` with some added functionality around
convenient simplifications for use in structural dynamics.

Basic usage
-----------

.. code-block:: python

    import sdypy as sd

    nodes = ... # Finite element nodes
    elements ... # Finite elements

    plotter = sd.view.Plotter3D(nodes, elements)
    plotter.add_fem_mesh(nodes, elements)
    plotter.show(show_grid=True, show_axes=True)

To animate a mode shape:

.. code-block:: python

    mode_shape = ... # Shape (n_nodes, 3)

    plotter = sd.view.Plotter3D(nodes, elements)
    plotter.add_fem_mesh(nodes, elements, animate=mode_shape)

To animate according to a time history:

.. code-block:: python

    time_history = ... # Shape (n_nodes, 3, n_timesteps)

    plotter = sd.view.Plotter3D(nodes, elements)
    plotter.add_fem_mesh(nodes, elements, animate=time_history)

To color the mesh according to a norm of the ``mode_shape``:

.. code-block:: python

    plotter = sd.view.Plotter3D(nodes, elements)
    plotter.add_fem_mesh(nodes, elements, animate=mode_shape, field="norm")

Here, if ``field="norm"`` is passed, the mesh will be colored according to the
norm of the ``mode_shape`` (or more generally, according to the norm of the ``animate`` argument).
To color the mesh according to an arbitrary field, the ``field`` argument must be of shape ``(n_nodes, n_timesteps)``. 

Animation
---------

The ``Plotter3D`` class also enables the user to animate the mesh. There are two way to animate the FEM mesh or points:

- By passing a 3D array of shape ``(n_nodes, 3, n_timesteps)`` to the ``animate`` argument. The points/nodes will be animated according to the time history.
- By passing a 2D array of shape ``(n_nodes, 3)`` to the ``animate`` argument. The ``animate`` defines the maximum displacement of the nodes. The animation is interpolated with a sine function.

The ``animate`` argument can also have the following shapes:

- ``(n_nodes*3)``: The ``Plotter3D`` will automatically convert this to ``(n_nodes, 3)``.
- ``(n_nodes*3, n_timesteps)``: The ``Plotter3D`` will automatically convert this to ``(n_nodes, 3, n_timesteps)``.
- ``(n_nodes*N)``: The ``Plotter3D`` will automatically convert this to ``(n_nodes, 3)``, where ``N`` is an arbitrary number. Only the first 3 columns are used. This is useful when animating e.g. a mode shape of a shell (each node has 6 DOF).
- ``(n_nodes*N, n_timesteps)``: The ``Plotter3D`` will automatically convert this to ``(n_nodes, 3, n_timesteps)``. Similar to previous point, but with time history.

The ``n_frames`` is set to 100 by default, but can be changed to alter the number of frames per one iteration. When providing the time history, the ``n_frames`` should match the last dimension of the ``animate`` argument.

Here is an example of animating a mode shape:

.. code-block:: python

    import sdypy as sd

    nodes = ... # Finite element nodes (n_nodes, 3)
    elements ... # Finite elements (n_elements, n_nodes_per_element)
    mode_shape = ... # Mode shape (n_nodes, 3)

    plotter = sd.view.Plotter3D(nodes, elements)
    plotter.add_fem_mesh(nodes, elements, animate=mode_shape, field="norm")

    plotter.start_animation(interval=10) # Required for animation. Interval is the time 1 frame is shown in ms
    plotter.add_animation_controls() # Optional, adds a button to start/stop the animation
    
    plotter.show()

Recording a GIF
---------------

To record a GIF, the ``gif_recorder`` method must be called. Example:

.. code-block:: python

    import sdypy as sd

    nodes = ... # Finite element nodes (n_nodes, 3)
    elements ... # Finite elements (n_elements, n_nodes_per_element)
    mode_shape = ... # Mode shape (n_nodes, 3)

    plotter = sd.view.Plotter3D(nodes, elements)
    
    plotter.gif_recorder("mode_shape.gif") # MUST BE CALLED BEFORE THE ANIMATION STARTS

    plotter.add_fem_mesh(nodes, elements, animate=mode_shape, field="norm")

    plotter.start_animation(interval=10) # Required for animation. Interval is the time 1 frame is shown in ms
    
    plotter.show()

The recording will last for 1 iteration of the animation. It will start recording when the animation starts.

To first adjust the view of the object in the plotter, use the ``.add_animation_controls()`` instead of ``.start_animation()``. This will
allow you to first manually adjust the view and then start the animation. At the start of the animation, the recording will start.