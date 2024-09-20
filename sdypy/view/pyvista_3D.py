import pyvista as pv
import numpy as np
from pyvistaqt import BackgroundPlotter
from pyvista import BasePlotter
from PyQt5.QtCore import QTimer

import warnings

def prepare_animation_displacements(data, n_nodes=None, n_frames=None):
    """
    Prepare the input data for the animation.

    Parameters
    ----------
    data : ndarray
        The input data array. It can have the following shapes:
        - (n_nodes, 3, n_frames)
        - (n_nodes*3, n_frames)
        - (n_nodes, 3) - a varying sine wave is applied to get the animation.
        - (n_nodes*3) - a varying sine wave is applied to get the animation.
        - (n_nodes*<int between 3 and 6>, n_frames) - just the first 3 directions are taken (x, y, and z).
        - (n_nodes*<int between 3 and 6>) - just the first 3 directions are taken (x, y, and z) and the sine wave is applied.
    n_nodes : int, optional
        The number of nodes in the mesh. Required if `data` is 1D or 2D and the shape does not provide enough information.
    n_frames : int, optional
        The number of frames for the animation. If not provided, it is inferred from the input data or set to 100.

    Returns
    -------
    ndarray
        The prepared data array with shape (n_nodes, 3, n_frames).

    Raises
    ------
    ValueError
        If the input data shape is not recognized or if the provided `n_nodes` or `n_frames` do not match the data dimensions.

    Notes
    -----
    If `n_frames` is not provided and cannot be inferred from the input data, it defaults to 100.
    """
    default_n_frames = 100

    if data.ndim == 3:
        if n_nodes is not None and data.shape[0] != n_nodes:
            raise ValueError("The number of nodes in the data should match the number of nodes in the mesh.")
        if n_frames is not None and data.shape[2] != n_frames:
            raise ValueError("The number of frames in the data should match the number of frames provided.")
        return data

    if data.ndim == 2:
        if data.shape[1] in range(3, 7):
            # data is of shape (n_nodes, n_dof_per_node)
            pass
        elif n_nodes is None:
            raise ValueError("The number of nodes should be provided when `data` is 2D and the second axis is not in size 3, 4, 5 or 6.")
        elif data.shape[0] // n_nodes in range(3, 7):
            # data is of shape (n_nodes*n_dof_per_node, n_frames)
            data = data.reshape(n_nodes, -1, data.shape[1])[:, :3, :]
            return data
        else:
            raise ValueError("The data shape is not recognized.")
        
    elif data.ndim == 1:
        if n_nodes is None:
            raise ValueError("The number of nodes should be provided when `data` is 1D.")
        
        if data.shape[0] // n_nodes in range(3, 7):
            # data is of shape (n_nodes*n_dof_per_node)
            data = data.reshape(n_nodes, -1)[:, :3]
        else:
            raise ValueError("The data shape is not recognized. When `data` is 1D, the size should be `n_nodes * n_dof`, where `n_dof` must be 3, 4, 5 or 6.")

    if data.ndim == 2: # if the data is of shape (n_nodes, n_dof_per_node), the frames are applied to the last dimension
        if n_frames is None:
            n_frames = default_n_frames
        data = np.cos(np.linspace(0, 2*np.pi, n_frames)[None, None, :] -  np.angle(data[:, :, None])) * np.abs(data[:, :, None])
    
    return data

def prepare_animation_field(data, n_nodes=None, n_frames=None):
    """
    Prepare the input field data for the animation.

    Note: What here is called "field" is called "scalars" in pyVista.

    Parameters
    ----------
    data : ndarray
        The input data array. It can have the following shapes:
        - (n_nodes) - a varying sine wave is applied to get the animation.
        - (n_nodes, n_frames)
    n_nodes : int, optional
        The number of nodes in the mesh. Required if `data` is 1D.
    n_frames : int, optional
        The number of frames for the animation. If not provided, it is inferred from the input data or set to 100.

    Returns
    -------
    ndarray
        The prepared data array with shape (n_nodes, n_frames).

    Raises
    ------
    ValueError
        If the input data shape is not recognized or if the provided `n_nodes` or `n_frames` do not match the data dimensions.

    Notes
    -----
    If `n_frames` is not provided and cannot be inferred from the input data, it defaults to 100.
    """
    default_n_frames = 100

    if data.ndim == 1:
        if n_nodes is None:
            raise ValueError("The number of nodes should be provided when `data` is 1D.")
        
        if data.shape[0] == n_nodes:
            # data is of shape (n_nodes)
            pass
        else:
            data = data.reshape(n_nodes, -1)

            if n_frames is not None and data.shape[1] != n_frames:
                raise ValueError("The number of frames in the data should match the number of frames provided.")
        
    elif data.ndim == 2:
        if n_nodes is not None and data.shape[0] != n_nodes:
            raise ValueError("The number of nodes in the data should match the number of nodes in the mesh.")
        if n_frames is not None and data.shape[1] != n_frames:
            raise ValueError("The number of frames in the data should match the number of frames provided.")
    
    if data.ndim == 1:
        if n_frames is None:
            n_frames = default_n_frames

        data = np.cos(np.linspace(0, 2*np.pi, n_frames)[None, :]) * data[:, None]
    
    return data

def create_fem_mesh(nodes, elements):
    """Create a PyVista mesh from nodes and elements.
    
    Parameters
    ----------
    nodes : np.ndarray
        The nodal coordinates of the mesh. Shape (n_nodes, 3).
    elements : np.ndarray
        The element connectivity of the mesh. Shape (n_elements, n_nodes_per_element).
    """
    n_nodes_per_element = elements.shape[1]

    mesh = pv.PolyData(nodes)
    faces = np.hstack([np.full((elements.shape[0], 1), n_nodes_per_element), elements])
    faces = faces.flatten().astype(np.int64)
    mesh.faces = faces

    return mesh

class Plotter3D(BackgroundPlotter, BasePlotter):
    """A PyVista background plotter with some additional functionality."""
    def __init__(self, *args, **kwargs):
        
        self.recording_gif = False
        self.legend_required = False
        self.animation_data = []
        self.animation_started = False
        self.mesh_dict = {}
        self.mesh_actor_dict = {}
        super().__init__(*args, **kwargs)

    def gif_recorder(self, gif_file: str, loop: int = 0, fps: int = 30, optimize: bool = True):
        """Open the GIF recorder from the pyVista Plotter.
        
        This does not yet start the recording.
        
        Requires the ``imageio`` package to be installed.

        A GIF is created according to https://tutorial.pyvista.org/tutorial/03_figures/d_gif.html

        Parameters
        ----------
        gif_file : str
            The file path to save the GIF, must end in '.gif'.
        loop : int, optional
            The number of loops for the GIF. Default is 0, which means infinite loops.
        fps : int, optional
            The frames per second of the GIF. Default is 30.
        optimize : bool, optional
            Optimize the GIF by only saving the difference between frames. This
            is used in the ``subrectangles`` argument of the pyVista ``open_gif`` method.
        """
        self.recording_gif = True
        self.open_gif(gif_file, loop=loop, fps=fps, palettesize=optimize)

    def add_fem_mesh(self, 
                     nodes, 
                     elements, 
                     field=None, 
                     field_name="field", 
                     cmap="viridis", 
                     edge_color='black', 
                     opacity=1,
                     animate=None,
                     n_frames=100
                    ):
        """Add a finite element mesh to the plotter.
        
        Parameters
        ----------
        nodes : np.ndarray
            The nodal coordinates of the mesh. Shape (n_nodes, 3).
        elements : np.ndarray
            The element connectivity of the mesh. Shape (n_elements, n_nodes_per_element).
        field : np.ndarray or string, optional
            The field values to be plotted. Can be array or "norm" or None. If "norm",
            the actual values are computed from the ``animate`` argument. Shape (n_nodes,).
        field_name : str, optional
            The name of the field array.
        cmap : str, optional
            The colormap to be used.
        edge_color : str, optional
            The color of the mesh edges.
        opacity : float, optional
            The opacity of the mesh.
        animate : np.ndarray
            The displacements or mode shape to be animated. Shape (n_points, 3, n_frames) or
            (n_points, 3) for mode shape. The points and directions can also be flattened to (n_points*3, n_frames).
            If there are more than 3 DOFs per node, only the first 3 are considered.
            To start the animation, call the ``start_animation`` method.
        n_frames : int
            The number of frames in a single period of the animation.
        """
        mesh = create_fem_mesh(nodes, elements)
        self.mesh_dict[id(mesh)] = mesh

        if type(field) is np.ndarray:
            field = prepare_animation_field(field, n_nodes=nodes.shape[0], n_frames=n_frames)

        if animate is not None:
            displacements = prepare_animation_displacements(animate, n_nodes=nodes.shape[0], n_frames=n_frames)

            mesh.points = mesh.points + displacements[:, :, 0]
            
            if type(field) is str and field == 'norm':
                field = np.linalg.norm(displacements, axis=1)

            # if field_name is already in animation_data, add different field_name
            field_names = [anim_dict["field_name"] for anim_dict in self.animation_data]
            if field_name in field_names:
                field_name = field_name + f"_{len(field_names)}"

            self.animation_data.append({
                "mesh_id": id(mesh),
                "displacements": displacements,
                "n_frames": n_frames,
                "frame": 0,
                "initial_points": nodes.copy(),
                "field": field,
                "field_name": field_name,
            })
            

        if field is not None:
            mesh.point_data[field_name] = field[:, 0]
            actor = self.add_mesh(mesh, show_edges=True, scalars=field_name, cmap=cmap, edge_color=edge_color, opacity=opacity)
            actor.mapper.scalar_range = (np.min(field), np.max(field)) # Set the field range

        else:
            actor = self.add_mesh(mesh, show_edges=True, edge_color=edge_color, opacity=opacity)
        
        self.mesh_actor_dict[id(mesh)] = actor

        return mesh

    def closeEvent(self, evt):
        """Override the close event to stop the q_timer if it exists.
        
        After the QTimer is stopped, the plotter is closed as usual.
        """
        if hasattr(self, "q_timer"):
            self.q_timer.stop()

        return super().closeEvent(evt)

    def add_fem_mode_shape(self, nodes, elements, mode_shape, cmap="viridis", edge_color='black', opacity=1, animate=False):
        """Add a mode shape to the plotter.
        
        This function uses the ``add_fem_mesh`` method to plot the mode shape. The nodes are moved according to the
        mode shape before plotting.

        Parameters
        ----------
        nodes : np.ndarray
            The nodal coordinates of the mesh. Shape (n_nodes, 3).
        elements : np.ndarray
            The element connectivity of the mesh. Shape (n_elements, n_nodes_per_element).
        mode_shape : np.ndarray
            The mode shape to be plotted. If 2 dimensional, it must be (n_nodes, 3). If
            1 dimensional, number of dofs per node is computed: ``len(mode_shape)//n_nodes``, then
            reshaped and first 3 dofs are considered x, y and z: ``mode_shape.reshape(-1, n_dof_per_node)[:, :3]``. 
        cmap : str, optional
            The colormap to be used.
        edge_color : str, optional
            The color of the mesh edges.
        opacity : float, optional
            The opacity of the mesh.
        animate : bool, optional
            Whether to animate the mode shape. Default is False.
        """
        if mode_shape.ndim == 1:
            n_dof_per_node = mode_shape.shape[0] // nodes.shape[0]
            mode_shape = mode_shape.reshape(-1, n_dof_per_node)[:, :3]
        
        if animate:
            mesh = self.add_fem_mesh(nodes, elements, scalar='norm', scalar_name="mode_shape", cmap=cmap, edge_color=edge_color, opacity=opacity, animate=mode_shape)
        else:
            nodes = nodes + mode_shape
            mesh = self.add_fem_mesh(nodes, elements, scalar=mode_shape, scalar_name="mode_shape", cmap=cmap, edge_color=edge_color, opacity=opacity)
        return mesh

    def start_animation(self, interval=10, blocking=False):
        """Start the animation.
        
        Parameters
        ----------
        interval : int, optional
            The interval between frames in milliseconds. Default is 100.
        blocking : bool, optional
            Whether the animation should be blocking. Default
        """
        if not hasattr(self, "timer"):
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_meshes)

        self.timer.start(interval)
        self.animation_started = True
        
        if blocking:
            self.app.exec_()

    def pause_animation(self):
        """Pause the animation."""
        self.timer.stop()
        self.animation_started = False

    def add_animation_controls(self, mode='button', interval=10, blocking=False, hotkey="space"):
        """Add a checkbox that starts and stops the animation.
        
        Parameters
        ----------
        mode : str, optional
            The mode of the animation controls. Can be 'button' or "hotkey". 
            If "button", a button is added to the plotter that toggles the animation.
            If "hotkey", a hotkey is used to toggle the animation. Default is 'button'.
        interval : int, optional
            The interval between frames in milliseconds. Default is 100.
        blocking : bool, optional
            Whether the animation should be blocking. Default is False.
        hotkey : str, optional
            The hotkey to be used if mode is "hotkey". Default is "space".
        """
        self.interval = interval
        self.blocking = blocking
        if mode == 'button':
            self.add_checkbox_button_widget(self.animation_callback, value=hasattr(self, "timer"))
        elif mode == 'hotkey':
            self.add_key_event(hotkey, lambda: self.animation_callback(True))
    
    def animation_callback(self, value):
        """The callback function that toggles the animation.

        Is used by the checkbox button widget or the key event.
        
        Parameters
        ----------
        value : bool
            The value of the checkbox. This doesn't have any effect, but the 
            value is expected when the checkbox is clicked.
        """
        if not self.animation_started:
            self.start_animation(self.interval, blocking=self.blocking)
        else:
            self.pause_animation()

    def _update_meshes(self):
        for anim_dict in self.animation_data:
            # self.update_mesh(**anim_dict)
            displacements = anim_dict["displacements"]
            n_frames = anim_dict["n_frames"]
            frame = anim_dict["frame"]
            initial_points = anim_dict["initial_points"]
            mesh = self.mesh_dict[anim_dict["mesh_id"]]
            field = anim_dict["field"]

            n_frames = displacements.shape[0]
            if frame >= n_frames or frame > displacements.shape[-1]-1:  # Loop the animation if desired, or stop
                if self.recording_gif:
                    # close the plotter and gif
                    self.close()
                    return False

                frame = 0

            # Update the mesh points with the current frame's data
            mesh.points = initial_points + displacements[:, :, frame]

            # update the field if provided
            if field is not None:
                field_name = anim_dict['field_name']
                mesh[field_name][:] = field[:, frame]
            
            self.render()  # Render the updated mesh
            
            frame += 1  # Move to the next frame

            anim_dict["frame"] = frame
        
        if self.recording_gif:
            self.write_frame()

    def add_points(self, points, color='red', point_size=5.0, render_points_as_spheres=False, label="", animate=None, n_frames=100, field=None, field_name="field", cmap="viridis", opacity=1):
        """Add points to the plotter.
        
        Parameters
        ----------
        points : np.ndarray
            The coordinates of the points. Shape (n_points, 3) or (3,) for a single point.
        color : str, optional
            The color of the points.
        point_size : float, optional
            The size of the points.
        render_points_as_spheres : bool, optional
            Whether to render the points as spheres. For a large number of points, 
            rendering as spheres can be slow. Default is False.
        label : str, optional
            The label of the points.
        animate : np.ndarray
            The displacements to be animated. Shape (n_points, 3, n_frames) or
            (n_points, 3) for mode shape. The points and directions can also be flattened to (n_points*3, n_frames).
            If there are more than 3 DOFs per node, only the first 3 are considered.
            To start the animation, call the ``start_animation`` method.
        n_frames : int
            The number of frames in a single period of the animation.
        field : np.ndarray or string, optional
            The field values to be plotted. Can be array or "norm" or None. If "norm",
            the actual values are computed from the ``animate`` argument. Shape (n_points,).
        field_name : str, optional
            The name of the field array.
        cmap : str, optional
            The colormap to be used.
        opacity : float, optional
            The opacity of the points.
        """
        if points.ndim == 1:
            points = points[None, :]

        mesh = pv.PolyData(points)
        self.mesh_dict[id(mesh)] = mesh

        if type(field) is np.ndarray:
            field = prepare_animation_field(field, n_nodes=points.shape[0], n_frames=n_frames)

        if render_points_as_spheres:
            mesh = mesh.glyph(scale=False, geom=pv.Sphere(radius=point_size/1000))


        if animate is not None:
            displacements = prepare_animation_displacements(animate, n_nodes=points.shape[0], n_frames=n_frames)
            
            if field == 'norm':
                field = np.linalg.norm(displacements, axis=1)

            # if field_name is already in animation_data, add different field_name
            field_names = [anim_dict["field_name"] for anim_dict in self.animation_data]
            if field_name in field_names:
                field_name = field_name + f"_{len(field_names)}"

            self.animation_data.append({
                "mesh_id": id(mesh),
                "displacements": displacements,
                "n_frames": n_frames,
                "frame": 0,
                "initial_points": points.copy(),
                "field": field,
                "field_name": field_name,
            })

            field_0 = field[:, 0] if field is not None else None
            

        if field is not None:
            mesh.point_data[field_name] = field_0
            actor = self.add_mesh(mesh, show_edges=True, scalars=field_name, cmap=cmap, opacity=opacity)
            actor.mapper.scalar_range = (np.min(field), np.max(field)) # Set the field range

        else:
            # actor = self.add_mesh(mesh, show_edges=True, edge_color=edge_color, opacity=opacity)
            actor = self.add_mesh(mesh, color=color, point_size=point_size, label=label, opacity=opacity)
        
        self.mesh_actor_dict[id(mesh)] = actor

        if label:
            self.legend_required = True

        return mesh

    def add_arrow(self, start, direction, color="black", scale=1, label="", **kwargs):
        """Add an arrow to the plotter.

        Parameters
        ----------
        start : np.ndarray
            The starting point of the arrow. Shape (3,).
        direction : np.ndarray
            The direction of the arrow. Shape (3,).
        color : str, optional
            The color of the arrow.
        scale : float, optional
            The scale of the arrow.
        label : str, optional
            The label of the arrow.
        
        Other Parameters
        ----------------
        kwargs : dict
            Additional keyword arguments to be passed to the pv.Arrow constructor.
        """
        arrow = pv.Arrow(start=start, direction=direction, scale=scale, **kwargs)
        self.mesh_dict[id(arrow)] = arrow

        actor = self.add_mesh(arrow, color=color, label=label)
        self.mesh_actor_dict[id(arrow)] = actor

        if label:
            self.legend_required = True

        return arrow

    def add_point_picker(self, callback=None):
        """Enable point picking on the plotter.
        
        Parameters
        ----------
        callback : callable, optional
            A callback function that will be called when a point is picked.
            The callback function should accept a single argument, which is the coordinates of the picked point.
            If no callback is provided, the picked point coordinates will be printed to the console and
            stored in the 'selected_points' attribute of the plotter.
        """
        if callback is None:
            print("The selected point coordinates are available in the 'selected_points' attribute.")
            self.selected_points = []
            def callback_function_point(point):
                self.selected_points.append(point)
                print(f"Selected point: {point}")
                self.add_mesh(pv.Sphere(radius=2, center=point), color='red')
        else:
            callback_function_point = callback

        self.enable_point_picking(callback_function_point)
    
    def show(self, show_grid=False, show_axes=False, bounding_box=False):
        """Show the plotter.
        
        Parameters
        ----------
        show_grid : bool, optional
            Whether to show the grid. Default is False.
        show_axes : bool, optional
            Whether to show the axes. Default is False.
        bounding_box : bool, optional
            Whether to add a bounding box. Default is False
        """
        try:
            if self.legend_required:
                self.add_legend()
        except:
            warnings.warn("Failed to add legend")

        try:
            if show_grid:
                self.show_grid()
        except:
            warnings.warn("Failed to show grid")
        
        try:
            if show_axes:
                self.show_axes()
        except:
            warnings.warn("Failed to show axes. If you use animation, the `show` method should be called before `start_animation`, or non-blocking `start_animation` should be used.")
        
        try:
            if bounding_box:
                self.add_bounding_box()
        except:
            warnings.warn("Failed to add bounding box")

        super().show()
