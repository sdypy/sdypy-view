SDyPy-view project
==================

The ``Plotter3D`` is a wrapper around the ``pyvista.BackgroundPlotter`` with some added functionality around
convenient simplifications for use in structural dynamics.

Notes
-----

**Running on Linux**

If the viewer does not open, maybe ``libxcb-cursor0`` is missing. Try installing (Ubuntu):

.. code-block:: shell

   sudo apt install libxcb-cursor0

Also, if running wayland, the following should be added at the top of the file:

.. code-block:: python

   os.environ['QT_QPA_PLATFORM'] = 'xcb'

**Running from shell**

If not running from jupyter notebook, the viewer might close right away. To keep it interactive, run it with:

.. code-block:: shell

   python -i file.py

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

    plotter.configure_animation(interval=20) # To adjust the interval and camera position
    plotter.start_animation() # Optional. To immediately start the animation. Otherwise, the controls are available in the plotter.
    plotter.show()

Recording a GIF
---------------

To record a GIF, the ``configure_gif_recorder`` method must be called. Example:

.. code-block:: python

    import sdypy as sd

    nodes = ... # Finite element nodes (n_nodes, 3)
    elements ... # Finite elements (n_elements, n_nodes_per_element)
    mode_shape = ... # Mode shape (n_nodes, 3)

    plotter = sd.view.Plotter3D(nodes, elements)
    
    plotter.gif_recorder("mode_shape.gif") # MUST BE CALLED BEFORE THE ANIMATION STARTS

    plotter.add_fem_mesh(nodes, elements, animate=mode_shape, field="norm")

    plotter.configure_gif_recorder('mode_shape.gif', fps=30) # Configure the GIF recorder
    
    plotter.show()

The recording will start on pressing the "Record" button in the toolbar. Alternatively, the
``configure_gif_recorder`` can be called like this:

.. code-block:: python

    plotter.configure_gif_recorder('mode_shape.gif', fps=30, start_on_play=True)

This will start the recording when the animation starts. To start the animation, call the ``start_animation`` method
or press the "Play" button in the toolbar.

The recording will last for 1 iteration of the animation.

Adding custom toolbar buttons
-----------------------------

To add custom toolbar commands, use the ``configure_toolbar`` method. Example:

.. code-block:: python

    import sdypy as sd

    nodes = ... # Finite element nodes (n_nodes, 3)
    elements ... # Finite elements (n_elements, n_nodes_per_element)
    mode_shape = ... # Mode shape (n_nodes, 3)

    plotter = sd.view.Plotter3D(nodes, elements)
    plotter.add_fem_mesh(nodes, elements, animate=mode_shape, field="norm")

    def custom_command():
        print("Custom command")

    toolbar_actions = {
        "Custom command": custom_command
    }

    plotter.configure_toolbar(toolbar_actions)

    plotter.show()
