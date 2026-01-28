import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import sdypy as sd
import meshio
import numpy as np
import sys

# Diagnostic information
print(f"Python version: {sys.version}")
print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")

try:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QAction
    print("✓ PyQt6 imported successfully")
except ImportError as e:
    print(f"✗ PyQt6 import failed: {e}")

try:
    import pyvistaqt
    print(f"✓ pyvistaqt version: {pyvistaqt.__version__}")
    try:
        print(f"  pyvistaqt Qt backend: {pyvistaqt.QtBinding}")
    except:
        pass
except Exception as e:
    print(f"✗ pyvistaqt issue: {e}")

try:
    mesh = meshio.read("examples/L_bracket.msh")
except:
    mesh = meshio.read("L_bracket.msh")


# extract nodes and elements from mesh
nodes = mesh.points
elements = []
for cells in mesh.cells:
    if cells.type == "quad":
        elements.append(cells.data)
elements = np.vstack(elements)


mode_shape = np.zeros((nodes.shape[0], 3))
mode_shape[:, 0] = 10.0

print("\n--- Creating Plotter3D ---")
try:
    plotter = sd.view.Plotter3D()
    print("✓ Plotter3D created successfully")
    print(f"  Plotter type: {type(plotter)}")
    print(f"  Has app_window: {hasattr(plotter, 'app_window')}")
    print(f"  Has app: {hasattr(plotter, 'app')}")

    plotter.add_fem_mesh(nodes, elements, animate=mode_shape, field="norm")
    print("✓ Mesh added successfully")

    print("\n--- Calling show() ---")
    plotter.show(show_grid=True, show_axes=True)
    print("✓ show() completed")

    # Check if window is visible
    if hasattr(plotter, 'app_window'):
        print(f"  Window visible: {plotter.app_window.isVisible()}")
        print(f"  Window size: {plotter.app_window.size()}")

except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()



# %%

