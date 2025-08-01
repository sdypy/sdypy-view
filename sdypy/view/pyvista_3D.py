import pyvista as pv
import numpy as np
from pyvistaqt import BackgroundPlotter
from pyvista import BasePlotter
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import  QAction
import pyperclip
from PIL import Image
import io
import platform
import subprocess

import warnings
from pyvista.plotting.picking import PICKED_REPRESENTATION_NAMES

def copy_image_to_clipboard(image: Image.Image):
    """Copy a PIL image to the clipboard on different platforms."""
    # Save image as PNG in a byte buffer
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()

    system = platform.system()

    # Windows: Using win32clipboard
    if system == "Windows":
        import win32clipboard
        from io import BytesIO
        
        # Convert to DIB format (Device Independent Bitmap)
        output = BytesIO()
        image.convert('RGB').save(output, 'BMP')
        data = output.getvalue()[14:]  # Remove BMP header
        output.close()
        
        # Copy to clipboard
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()

    # macOS
    elif system == "Darwin":
        process = subprocess.Popen(['osascript', '-e', 'set the clipboard to (read (POSIX file "/dev/stdin") as TIFF picture)'],
                                stdin=subprocess.PIPE)
        process.communicate(input=image_bytes)

    # Linux
    elif system == "Linux":
        subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png'], input=image_bytes)

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
        data = np.sin(np.linspace(0, 2*np.pi, n_frames)[None, None, :] -  np.angle(data[:, :, None])) * np.abs(data[:, :, None]) # mora biti sinus
    
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

        data = np.sin(np.linspace(0, 2*np.pi, n_frames)[None, :]) * data[:, None]
    
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
        self._accel_widgets = []        # List of widget objects
        self._accel_widget_data = []   # List of widget data dicts
        super().__init__(*args, **kwargs)

        # Default settings
        self.interval = 10
        self.blocking = False
        self.animate = None
        self.n_frames = 100
        self.default_accelerometer_size = (0.1, 0.1, 0.05)
        self.default_excitation_size = 1
        self.snap_to_closest_node =True

        self.app_window.addToolBarBreak() # add a break to the toolbar
        # Custom toolbar and menu items

        self.animation_toolbar = self.app_window.addToolBar('SDyPy Toolbar')
        self.add_toolbar_action(self.animation_toolbar, "Screenshot to clipboard", self.copy_screenshot_to_clipboard, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Play", self.start_animation, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Pause", self.pause_animation, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Stop", self.reset_animation, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Record GIF", self.start_recording, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Add accelerometer", self.add_accelerometer_widget, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Pick a point", self.add_point_picker, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Print cell type", self.print_cell_type_on_click, self.app_window)
        self.add_toolbar_action(self.animation_toolbar, "Add excitation", self.add_excitation, self.app_window)

        menu = self.main_menu.addMenu("Camera position")
        menu.addAction("Print current camera position", lambda: print(self.camera_position))
        menu.addAction("Copy current camera position", lambda: pyperclip.copy(self.camera_position))

    def copy_screenshot_to_clipboard(self):
        """Take a screenshot of the plotter and save it to the clipboard."""
        try:
            # Capture the screenshot
            image_array = self.screenshot(transparent_background=True)

            # Convert to PIL Image
            image = Image.fromarray(image_array)

            # Copy to clipboard using platform-specific methods
            copy_image_to_clipboard(image)

            print("Screenshot copied to clipboard!")
        except Exception as e:
            print(f"Failed to copy screenshot: {e}")

    def configure_toolbar(self, custom_actions=dict()):
        """Configure the toolbar of the plotter.
        
        Custom actions can be added in a form of a dictionary with the action name as the key and the action method as the value.
        The action (function) should not take any arguments.

        https://github.com/pyvista/pyvista-support/issues/122

        Example:

        .. code-block:: python

            def custom_action():
                print("Custom action")
            
            custom_actions = {"My action": custom_action}

        Parameters
        ----------
        custom_actions : dict
            A dictionary with the action name as the key and the action method as the value.
        """
        # Add a toolbar
        self.app_window.addToolBarBreak() # add a break to the toolbar
        user_toolbar = self.app_window.addToolBar('User Toolbar')

        for key, method in custom_actions.items():
            self.add_toolbar_action(user_toolbar, key, method, self.app_window)

    def add_toolbar_action(self, toolbar, key, method, main_window):
        action = QAction(key, main_window)
        action.triggered.connect(method)
        toolbar.addAction(action)
        return

    def add_fem_mesh(self, 
                     nodes, 
                     elements, 
                     field=None, 
                     field_name="field", 
                     cmap="viridis", 
                     edge_color='black', 
                     opacity=1,
                     animate=None,
                     n_frames=None
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

        if animate is None:
            animate = self.animate
        if n_frames is None:
            n_frames = self.n_frames
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

        return actor

    def closeEvent(self, evt):
        """Override the close event to stop the ``timer`` if it exists.
        
        After the QTimer is stopped, the plotter is closed as usual.
        """
        if hasattr(self, "timer"):
            self.timer.stop()
            self.timer.deleteLater()

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

       # spremenil scalar v field in scalar_ name v field_name 
        if animate:
            actor = self.add_fem_mesh(nodes, elements, field='norm', field_name="mode_shape", cmap=cmap, edge_color=edge_color, opacity=opacity, animate=mode_shape)
        else:
            nodes = nodes + mode_shape
            actor = self.add_fem_mesh(nodes, elements, field=mode_shape, field_name="mode_shape", cmap=cmap, edge_color=edge_color, opacity=opacity)
        return actor

    def start_animation(self):
        """Start the animation.
        
        Parameters
        ----------
        interval : int, optional
            The interval between frames in milliseconds. Default is 100.
        blocking : bool, optional
            Whether the animation should be blocking. Default
        """
        if self.animation_started or not self.animation_data:
            return
        
        if not hasattr(self, "timer"):
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_meshes)

        self.timer.start(self.interval)
        self.animation_started = True
        
        self.place_accelerometer_from_widget(animate=self.animate, n_frames=self.n_frames)
        self.start_animation()

        # Remove line widgets when animation starts
        if hasattr(self, 'line_widgets'):
            for widget in self.line_widgets:
                try:
                    widget.Off()  # Turn off the widget
                    widget.SetEnabled(False)  # Disable the widget
                except Exception:
                    pass
    
        # Odstrani osvetlitev
        for name in PICKED_REPRESENTATION_NAMES.values():
            self.remove_actor(name)

        if hasattr(self, "_displacement_preview_actor"):
            self.remove_actor(self._displacement_preview_actor) 
            del self._displacement_preview_actor
        if self.blocking:
            self.app.exec_()

    def pause_animation(self):
        """Pause the animation."""
        if not self.animation_started:
            return
        
        self.timer.stop()
        self.animation_started = False

    def reset_animation(self):
        """Reset the frames to 0 and update the meshes."""
        self.pause_animation()

        self._update_meshes(frame=0) # update the meshes to the initial frame

    def start_recording(self):
        """Start the recording of the animation."""
        self.reset_animation()
        self.recording_gif = True
        self.start_animation()

    def configure_animation(self, interval=10, camera_position=None, blocking=False):
        """Configure the animation settings.
        
        Parameters
        ----------
        interval : int, optional
            The interval between frames in milliseconds. Default is 100.
        camera_position : str, optional
            The camera position to be used for the animation. If not provided, the current camera position is used.
        blocking : bool, optional
            Whether the animation should be blocking. Default is False.
        """
        if camera_position is not None:
            self.camera_position = camera_position

        self.interval = interval
        self.blocking = blocking

    def configure_gif_recorder(self, gif_file: str, loop: int = 0, fps: int = 30, start_on_play: bool = False, optimize: bool = True):
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
        start_on_play : bool, optional
            If True, start the recording when the play button is pressed. If False,
            the recording is started by pressing "Record" button. Default is False.
        optimize : bool, optional
            Optimize the GIF by only saving the difference between frames. This
            is used in the ``subrectangles`` argument of the pyVista ``open_gif`` method.
        """
        self.add_toolbar_action(self.animation_toolbar, "Record", self.start_recording, self.app_window)
        if start_on_play:
            self.recording_gif = True
        else:    
            self.recording_gif = False
        self.open_gif(gif_file, loop=loop, fps=fps, palettesize=optimize)

    def _update_meshes(self, frame=None):
        """Update the meshes for the animation.
        
        Parameters
        ----------
        frame : int, optional
            The frame to be updated. If not provided, the frame is taken from the animation data.
        """
        for anim_dict in self.animation_data:
            if frame is None:
                frame = anim_dict["frame"]
            
            displacements = anim_dict["displacements"]
            n_frames = anim_dict["n_frames"]
            initial_points = anim_dict["initial_points"]
            mesh = self.mesh_dict[anim_dict["mesh_id"]]
            field = anim_dict["field"]

            if frame >= n_frames or frame > displacements.shape[-1]-1:  # Loop the animation if desired, or stop
                if self.recording_gif:
                    # close the plotter and gif
                    self.add_text("Saving GIF, please wait...")
                    self.close()
                    return False

                frame = 0

            # Update the mesh points with the current frame's data
            mesh.points = initial_points + displacements[:, :, frame]

            # update the field if provided
            if field is not None:
                field_name = anim_dict['field_name']
                mesh[field_name][:] = field[:, frame]
            
            
            frame += 1  # Move to the next frame

            anim_dict["frame"] = frame

        self.render()  # Render the updated mesh

        if self.recording_gif:
            self.write_frame()

    def add_points(self, points, color='red', point_size=5.0, render_points_as_spheres=False, label="", 
                  animate=None, n_frames=None, field=None, field_name="field", cmap="viridis", opacity=1,
                  connect_points=False, line_width=1.0):
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
        connect_points : bool or list, optional
            If True, connects points sequentially with lines. If a list, should contain
            pairs of point indices to connect specific points. Default is False.
        line_width : float, optional
            Width of the connecting lines. Default is 1.0.
        """
        if points.ndim == 1:
            points = points[None, :]
        if animate is None:
            animate = self.animate
        if n_frames is None:
            n_frames = self.n_frames

        mesh = pv.PolyData(points)
        
        if connect_points is not False:
            if connect_points is True:
                lines = np.column_stack((
                    np.full(points.shape[0]-1, 2), 
                    np.arange(points.shape[0]-1),    
                    np.arange(1, points.shape[0])    
                )).ravel()
            else:
              
                lines = np.column_stack((
                    np.full(len(connect_points), 2),  
                    [pair[0] for pair in connect_points],
                    [pair[1] for pair in connect_points]   
                )).ravel()
            mesh.lines = lines

        self.mesh_dict[id(mesh)] = mesh

        if type(field) is np.ndarray:
            field = prepare_animation_field(field, n_nodes=points.shape[0], n_frames=n_frames)

        if render_points_as_spheres:
            mesh = mesh.glyph(scale=False, geom=pv.Sphere(radius=point_size/1000))


        if animate is not None:
            displacements = prepare_animation_displacements(animate, n_nodes=points.shape[0], n_frames=n_frames)

            mesh.points = mesh.points + displacements[:, :, 0]
            
            if field == 'norm':
                field = np.linalg.norm(displacements, axis=1)

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
            actor = self.add_mesh(mesh, show_edges=True, scalars=field_name, cmap=cmap, 
                                opacity=opacity, line_width=line_width)
            actor.mapper.scalar_range = (np.min(field), np.max(field))

        else:
            actor = self.add_mesh(mesh, color=color, point_size=point_size, label=label, 
                                opacity=opacity, line_width=line_width)
        
        self.mesh_actor_dict[id(mesh)] = actor

        if label:
            self.legend_required = True

        return actor
    
    def add_surface(self, points, color='red', point_size=5.0, render_points_as_spheres=False, label="", animate=None, n_frames=None, field=None, field_name="field", cmap="viridis", opacity=1):
        """Add surface to the plotter.

        The surface is created by triangulation of the points using the ``delaunay_2d`` method.
        Because the ``delaunay_2d`` method is used, the points must be approximately coplanar.
        
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
        if animate is None:
            animate = self.animate
        if n_frames is None:
            n_frames = self.n_frames

        cloud = pv.PolyData(points)
        surface = cloud.delaunay_2d() 
        mesh = surface.extract_surface()
        self.mesh_dict[id(mesh)] = mesh

        if type(field) is np.ndarray:
            field = prepare_animation_field(field, n_nodes=points.shape[0], n_frames=n_frames)

        if render_points_as_spheres:
            mesh = mesh.glyph(scale=False, geom=pv.Sphere(radius=point_size/1000))


        if animate is not None:
            displacements = prepare_animation_displacements(animate, n_nodes=points.shape[0], n_frames=n_frames)

            mesh.points = mesh.points + displacements[:, :, 0]
            
            if field == 'norm':
                field = np.linalg.norm(displacements, axis=1)

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
            actor = self.add_mesh(mesh, color=color, point_size=point_size, label=label, opacity=opacity)
        
        self.mesh_actor_dict[id(mesh)] = actor

        if label:
            self.legend_required = True

        return actor

    def add_arrow(
        self,
        start,
        direction,
        color="black",
        scale=1,
        label="",
        animate=None,
        n_frames=None,
        **kwargs
    ):
        """
        Add an arrow to the plotter. If 'animate' is provided, the arrow will be animated.

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
        animate : np.ndarray, optional
            The displacements to animate the arrow. Shape (1, 3, n_frames) or (1, 3).
        n_frames : int, optional
            Number of animation frames.
        kwargs : dict
            Additional keyword arguments for pv.Arrow.

        Returns
        -------
        actor : pyvista.Actor
            The actor for the arrow.
        arrow_displacements : np.ndarray or None
            The animation displacements if animated, else None.
        """
        arrow = pv.Arrow(start=start, direction=direction, scale=scale, **kwargs)
        self.mesh_dict[id(arrow)] = arrow

        if animate is None:
            animate = self.animate
        if n_frames is None:
            n_frames = self.n_frames

        arrow_displacements = None
        if animate is not None:
            displacements = prepare_animation_displacements(animate, n_nodes=1, n_frames=n_frames)
            arrow_displacements = np.tile(displacements[0, :, :], (arrow.points.shape[0], 1, 1))

            self.animation_data.append({
                "mesh_id": id(arrow),
                "displacements": arrow_displacements,
                "n_frames": n_frames,
                "frame": 0,
                "initial_points": arrow.points.copy(),
                "field": None,
                "field_name": None,
            })

        actor = self.add_mesh(arrow, color=color, label=label, **kwargs)
        self.mesh_actor_dict[id(arrow)] = actor

        if label == "Normal vector":
            if not hasattr(self, "_accelerometer_arrow_ids"):
                self._accelerometer_arrow_ids = []
            self._accelerometer_arrow_ids.append(id(arrow))

        if label:
            self.legend_required = True

        return actor

    def add_point_picker(self, *,callback=None):
        """Enable point picking on the plotter.
        
        Parameters
        ----------
        callback : callable, optional
            A function to be called when a point is picked. The function should accept a single argument, which is the picked point coordinates.
            If not provided, the selected points will be stored in the `selected_points` attribute and printed to the console.
        """
        print("callback:", callback, type(callback))
        if callback is not None:
            callback_function_point = callback
        else:
            print("The selected point coordinates are available in the 'selected_points' attribute.")
            self.selected_points = []

            def callback_function_point(point):
                self.selected_points.append(point)
                print(f"Selected point: {point}")
                self.add_mesh(pv.Sphere(radius=0.02, center=point), color='red')

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


    def calculate_element_normal(self, element_points):
        """
        Calculate the normal vector of an element defined by its corner points.

        Parameters
        ----------
        element_points : array-like, tuple, or list
            An array of shape (n_points, 3) where n_points >= 3, representing the corner points of the element.

        Returns
        -------
        normal : np.ndarray
            The normal vector of the element, normalized to unit length.
        """
        element_points = np.asarray(element_points)

        if element_points.shape[0] < 3 or element_points.shape[1] != 3:
            raise ValueError("Need at least 3 points with 3 coordinates each to define an element normal.")

        vector1 = element_points[1] - element_points[0]
        vector2 = element_points[2] - element_points[0]

        normal = np.cross(vector1, vector2)
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude == 0:
            raise ValueError("The provided points are collinear or coincident, cannot compute a normal vector.")
        normal = normal / normal_magnitude

        camera_position = np.array(self.camera_position[0])
        camera_direction = np.array(self.camera_position[1])

        camera_normal = camera_direction - camera_position
        camera_normal = camera_normal / np.linalg.norm(camera_normal)
        if np.dot(normal, camera_normal) > 0:
            normal = -normal

        return normal
    

    def align_box_widget_bounds(self, center, size, normal):
        """
        Returns bounds for a box widget such that its local z-axis is aligned with the given normal.

        Parameters
        ----------
        center : array-like
            The center of the box.
        size : tuple of float
            The size of the box (x_length, y_length, z_length).
        normal : array-like
            The target normal vector to align the box's z-axis with.

        Returns
        -------
        bounds : list
            The bounds for the box widget, possibly rotated.
        """

        center = np.asarray(center)
        size = np.asarray(size)
        normal = np.asarray(normal)
        normal = normal / np.linalg.norm(normal)

        z_axis = np.array([0, 0, 1])

        axis = np.cross(z_axis, normal)
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

        if np.linalg.norm(axis) < 1e-6:
            if np.dot(z_axis, normal) < 0:
                rot_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
            else:
                rot_matrix = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rot_matrix = (
                np.eye(3) +
                np.sin(angle) * K +
                (1 - np.cos(angle)) * (K @ K)
            )

        dx, dy, dz = size / 2.0
        corners = np.array([
            [dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, dz],
            [dx, -dy, -dz],
            [-dx, dy, dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [-dx, -dy, -dz],
        ])

        rotated_corners = (rot_matrix @ corners.T).T + center

        xmin, ymin, zmin = rotated_corners.min(axis=0)
        xmax, ymax, zmax = rotated_corners.max(axis=0)
        bounds = [xmin, xmax, ymin, ymax, zmin, zmax]

        return bounds

    def add_accelerometer_widget(self, set_handle_size=0.005, cube_size=None, tolerance=0):
        """
        Place a cube widget at a picked point, allow user to adjust it, and store its state for animation.
        
        Parameters
        ----------
        snap_to_closest-node : bool, optional
            If True, the widget will snap to the closest node of the picked mesh. Default is True.
        set_handle_size : float, optional
            Size of the widget handles. Default is 0.005.
        cube_size : tuple of float, optional
            Size of the cube widget in (x, y, z) dimensions. Default is (0.1, 0.1, 0.05).
        tolerance : float, optional
            Tolerance for picking. Default is 0.
        """

        if cube_size is None:
            cube_size = self.default_accelerometer_size
        if not hasattr(self, "_accel_widgets"):
            self._accel_widgets = []
        if not hasattr(self, "_accel_widget_data"):
            self._accel_widget_data = []

        def on_pick(picked_point, picker): 
            mesh = picker.GetDataSet()
            if mesh is None:
                print("No mesh found at picked point.")
                return

            cell_id = mesh.find_closest_cell(picked_point)
            if cell_id < 0:
                print("No cell found at picked point.")
                return

            cell = mesh.extract_cells(cell_id)
            points = cell.points

            if points.shape[0] >= 3:
                normal = self.calculate_element_normal([points[0], points[1], points[2]])
            else:
                print("Cell does not have enough points to calculate normal.")
                return

            self.disable_picking()
            if self.snap_to_closest_node:
                node_id = np.argmin(np.linalg.norm(points - picked_point, axis=1))
                center = points[node_id]
            else:
                center = picked_point

            offset = 0.5 * cube_size[2] * normal
            cube_center = center + offset
            bounds = self.align_box_widget_bounds(cube_center, cube_size, normal)

            def widget_callback(box_polydata, widget=None):
                bounds = box_polydata.bounds
                center = box_polydata.center

                try:
                    idx = self._accel_widgets.index(widget)
                except ValueError:
                    idx = -1

                data = {
                    "bounds": bounds,
                    "center": center,
                    "polydata": box_polydata.copy(),
                    "size": (bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]),
                }

                if 0 <= idx < len(self._accel_widget_data):
                    self._accel_widget_data[idx] = data
                else:
                    self._accel_widget_data.append(data)

            widget = self.add_box_widget(
                callback=widget_callback,
                bounds=bounds,
                rotation_enabled=True,
                color="red",
                outline_translation=True,
                pass_widget=True
            )
            try:
                widget.SetHandleSize(set_handle_size)
            except Exception:
                pass
            self._accel_widgets.append(widget)

        self.enable_surface_point_picking(callback=on_pick, use_picker=True, tolerance=tolerance)

    def place_accelerometer_from_widget(self, animate=None, n_frames=None):
        """
        Place the animated accelerometer at the last widget's position and orientation.

        Parameters
        ----------
        animate : ndarray, optional
            Displacement matrix of shape (1, 3) that defines the direction of animation.
            If None, uses the default animation settings.
        n_frames : int, optional
            Number of animation steps. If None, uses the default number of frames.
        """

        if animate is None:
            animate = self.animate
        if n_frames is None:
            n_frames = self.n_frames

        # if not hasattr(self, "_accel_widget_data") or not self._accel_widget_data:
        #     print("No accelerometer widget placed.")
        #     return

        # Remove all widgets from the scene
        if hasattr(self, "_accel_widgets"):
            for widget in self._accel_widgets:
                try:
                    widget.SetEnabled(False)
                    widget.Off()
                except Exception:
                    pass
            self._accel_widgets = []

        for data in self._accel_widget_data:
            center = data["center"]
            size = data["size"]
            self.add_cube(
                center=center,
                size=size,
                color="red",
                label="Accelerometer",
                animate=animate,
                n_frames=n_frames,
                center_to_normal=center,
            )
        self._accel_widget_data = []
        
    def add_cube(
        self,
        center,
        size=(1, 1, 1),
        color="red",
        label="",
        style='surface',
        animate=None,
        n_frames=None,
        normal=None,
        center_to_normal=None,
        **kwargs
    ):
        """
        Add an rectangular cube to the plotter.

        Parameters
        ----------
        center : array-like
            The center of the cube.
        size : tuple of float, optional
            The dimensions of the cube in (x, y, z) directions. Default is (0.1, 0.1, 0.05).
        color : str, optional
            The color of the cube. Default is "red".
        label : str, optional
            The label for the legend.
        style : str, optional
            The rendering style for the cube mesh (e.g., 'surface').
        animate : ndarray, optional
            Displacement matrix of shape (1, 3) or (1, 3, n_frames) that defines the direction and magnitude of animation.
        n_frames : int, optional
            Number of animation steps. If None, uses the default number of frames.
        normal : array-like, optional
            The normal vector to align the cube's local z-axis.
        center_to_normal : array-like, optional
            The point about which to align the cube to the normal.
        **kwargs : dict
            Additional keyword arguments for `add_mesh`.

        Returns
        -------
        actor : pyvista.Actor
            The actor for the cube.
        cube_displacements : np.ndarray or None
            The animation displacements if animated, else None.
        """
        cube = pv.Cube(
            center=center,
            x_length=size[0],
            y_length=size[1],
            z_length=size[2]
        )
        if animate is None:
            animate = self.animate
        if n_frames is None:
            n_frames = self.n_frames
        if normal is not None and center_to_normal is not None:
            cube = self.align_cube_to_normal(cube, normal, center_to_normal)

        self.mesh_dict[id(cube)] = cube

        if animate is not None and animate.ndim == 2:
            displacements = prepare_animation_displacements( 
                animate, n_nodes=1, n_frames=n_frames   #n_nodes=1 zato ker gledamo samo eno točko
            )
            cube_displacements = np.tile(displacements[0, :, :], (cube.points.shape[0], 1, 1))

            self.animation_data.append({
                "mesh_id": id(cube),
                "displacements": cube_displacements,
                "n_frames": n_frames,
                "frame": 0,
                "initial_points": cube.points.copy(),
                "field": None,
                "field_name": None,
            })

        actor = self.add_mesh(cube, color=color, label=label, style=style, **kwargs)
        self.mesh_actor_dict[id(cube)] = actor

        if label:
            self.legend_required = True

        return actor

    def align_cube_to_normal(self, cube, normal, center):
        """
        Rotates the cube so its local z-axis aligns with the given normal vector,
        using 'center' as the rotation point.

        Parameters
        ----------
        cube : pyvista.PolyData
            The cube mesh to rotate.
        normal : array-like
            The target normal vector to align the cube's z-axis with.
        center : array-like
            The point about which to rotate the cube.

        Returns
        -------
        pyvista.PolyData
            The rotated cube mesh.
        """
        normal = np.asarray(normal)
        normal = normal / np.linalg.norm(normal)

        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, normal)
        angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0)))

        if np.linalg.norm(axis) < 1e-6:
            if np.dot(z_axis, normal) < 0:
                # rotate 180 degrees
                return cube.rotate_x(180, point=center, inplace=False)
            else:
                # Already aligned
                return cube
        else:
            return cube.rotate_vector(axis, angle, point=center, inplace=False)


    def add_cube_widget(
        self,
        callback,
        center=(0, 0, 0),
        size=(1, 1, 1),
        color="red",
        outline_color="red",
        rotation_enabled=True,
        outline_translation=True,
        pass_widget=False,
        test_callback=True,
        interaction_event='end',
    ):
        """
        Add an interactive cube (box) widget to the scene.

        Parameters
        ----------
        callback : callable
            Function called when the cube is moved or resized. Receives the box PolyData or planes.
        center : tuple, optional
            Center of the cube.
        size : float or tuple, optional
            Size of the cube (float for isotropic, tuple for (x, y, z)).
        color : str or tuple, optional
            Color of the box faces.
        outline_color : str or tuple, optional
            Color of the box outline.
        rotation_enabled : bool, optional
            Allow rotation of the box.
        outline_translation : bool, optional
            Allow translation of the box.
        pass_widget : bool, optional
            If True, pass the widget object to the callback.
        test_callback : bool, optional
            If True, run the callback after widget creation.
        interaction_event : str, optional
            VTK interaction event to trigger the callback.

        Returns
        -------
        vtk.vtkBoxWidget
            The box widget.
        """

        center = np.array(center)
        size = np.asarray(size)
        half_size = size / 2.0
        bounds = [
            center[0] - half_size[0], center[0] + half_size[0],
            center[1] - half_size[1], center[1] + half_size[1],
            center[2] - half_size[2], center[2] + half_size[2],
        ]

        def _the_callback(box_polydata, widget=None):
            # box_polydata is a pyvista.PolyData representing the box
            if callable(callback):
                if pass_widget:
                    callback(box_polydata, widget)
                else:
                    callback(box_polydata)

        box_widget = self.add_box_widget(
            callback=_the_callback,
            bounds=bounds,
            rotation_enabled=rotation_enabled,
            color=color,
            outline_translation=outline_translation,
            pass_widget=pass_widget,
            interaction_event=interaction_event,
        )
        box_widget.GetOutlineProperty().SetColor(pv.Color(outline_color).float_rgb)

        if test_callback:
            the_box = pv.PolyData()
            box_widget.GetPolyData(the_box)
            _the_callback(the_box)

        return box_widget
    

    def print_cell_type_on_click(self, tolerance=0.1):
        """
        Add a toolbar action that prints the type of the cell (quad, triangle, vertex, etc.)
        when you click on a cell in the plotter.
        """

        def on_pick(picked_point, picker):
            mesh = picker.GetDataSet()
            if mesh is None:
                print("No mesh found at picked point.")
                return

            cell_id = mesh.find_closest_cell(picked_point)
            if cell_id < 0:
                print("No cell found at picked point.")
                return

            # For UnstructuredGrid and PolyData
            try:
                cell_type_id = mesh.celltypes[cell_id]
                cell_type_name = pv.CellType(cell_type_id).name.lower()
                print(type(mesh))
            except Exception:
                # For PolyData without celltypes
                cell = mesh.extract_cells(cell_id)
                n_points = cell.points.shape[0]
                if n_points == 4:
                    cell_type_name = "quad"
                elif n_points == 3:
                    cell_type_name = "triangle"
                elif n_points == 2:
                    cell_type_name = "line"
                elif n_points == 1:
                    cell_type_name = "vertex"
                else:
                    cell_type_name = f"polygon ({n_points} points)"
                print(type(mesh))
            print(f"Clicked cell ID: {cell_id}, type: {cell_type_name}")

        self.enable_surface_point_picking(callback=on_pick, use_picker=True, tolerance=tolerance)
    
    def add_excitation(self, tolerance=0):
        """
        Add an interactive arrow widget. First click on a surface to place the arrow start point,
        then use the line widget to adjust the arrow direction and length. The arrow initially
        points in the normal direction of the clicked surface. Use checkbox to reverse the arrow direction.
        
        Parameters
        ----------
        tolerance : float, optional
            Tolerance for surface picking. Default is 0.
        """
        if not hasattr(self, '_arrow_widget_actors'):
            self._arrow_widget_actors = []
        
        if not hasattr(self, '_arrow_checkbox_count'):
            self._arrow_checkbox_count = 0
        
        def on_surface_pick(picked_point, picker):            
            nonlocal self
            mesh = picker.GetDataSet()
            if mesh is None:
                print("No mesh found at picked point.")
                return

            cell_id = mesh.find_closest_cell(picked_point)
            if cell_id < 0:
                print("No cell found at picked point.")
                return

            cell = mesh.extract_cells(cell_id)
            points = cell.points

            if points.shape[0] >= 3:
                normal = self.calculate_element_normal([points[0], points[1], points[2]])
            else:
                print("Cell does not have enough points to calculate normal.")
                return

            arrow_start_point = np.asarray(picked_point)
            self.disable_picking()
            
            arrow_length = self.default_excitation_size 
            initial_end = arrow_start_point + normal * arrow_length

            current_arrow_actor = None
            current_line_widget = None
            is_reversed = False
            
            self._arrow_checkbox_count += 1
            checkbox_position = (10, 50 + (self._arrow_checkbox_count - 1) * 55)
            first_callback_called = False

            def line_callback(point1, point2):
                nonlocal current_arrow_actor, first_callback_called

                if not first_callback_called:
                    first_callback_called = True
                    return

                if current_arrow_actor is not None:
                    self.remove_actor(current_arrow_actor)
                    if current_arrow_actor in self._arrow_widget_actors:
                        self._arrow_widget_actors.remove(current_arrow_actor)
                
                start = np.asarray(point1)
                end = np.asarray(point2)
                direction = end - start
                length = np.linalg.norm(direction)
                
                if length > 0:
                    direction_normalized = direction / length
                else:
                    direction_normalized = normal 
                    length = 0.1
                
                if is_reversed:
                    arrow_start = end
                    arrow_direction = -direction_normalized
                else:
                    arrow_start = start
                    arrow_direction = direction_normalized

                current_arrow_actor = self.add_arrow(
                    start=arrow_start,
                    direction=arrow_direction,
                    color='red',
                    scale=length,
                    label=f"Interactive Arrow {self._arrow_checkbox_count}"
                )
                self._arrow_widget_actors.append(current_arrow_actor)

            current_line_widget = self.add_line_widget(
                callback=line_callback,  
                use_vertices=True,
                color='blue'
            )
            current_line_widget.SetHandleSize(0.005)
            current_line_widget.SetPoint1(arrow_start_point)
            current_line_widget.SetPoint2(initial_end)

            def reverse_arrow_callback(value):
                nonlocal is_reversed
                is_reversed = value
                
                if current_line_widget:
                    point1 = current_line_widget.GetPoint1()
                    point2 = current_line_widget.GetPoint2()
                    line_callback(point1, point2)
            
            self.add_checkbox_button_widget(
                reverse_arrow_callback, 
                value=False,
                position=checkbox_position,
                size=50
            )
            if self._arrow_checkbox_count == 1:
                self.add_text(
                    "Reverse Arrow",
                    position="lower_left",
                    color='black',
                    font_size=12
                )
            first_callback_called = True
            line_callback(arrow_start_point, initial_end)

        self.enable_surface_point_picking(callback=on_surface_pick, use_picker=True, tolerance=tolerance)
