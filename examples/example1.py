import sdypy as sd
import meshio
import numpy as np

mesh = meshio.read("./examples/L_bracket.msh")

nodes = mesh.points
elements = []
for cells in mesh.cells:
    if cells.type == "quad":
        elements.append(cells.data)

elements = np.vstack(elements)

mode_shape = np.zeros((nodes.shape[0], 3))
mode_shape[:, 0] = 10

plotter = sd.view.Plotter3D()
plotter.animate = mode_shape
plotter.n_frames = 400
plotter.default_accelerometer_size = np.array([10, 10, 5])
plotter.default_excitation_size = 50
plotter.snap_to_closest_node = True

plotter.add_fem_mesh(nodes, elements, animate=mode_shape, field=None)
plotter.add_axes()

import sys
from PyQt5.QtWidgets import QApplication
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
app.exec_()


