from spb.backends.base_backend import Plot
import numpy as np
from spb.utils import get_vertices_indices
from spb.defaults import pyvista_theme, pyvista_bg_color
import pyvista as pv
pv.set_plot_theme(pyvista_theme)

"""
TODO:
    1. Interactivity within jupyter notebook
    2. color maps - evaluate if possible to use same colormaps across multiple
        backends

"""

class PyvistaBackend(Plot):
    """ A backend for plotting SymPy's symbolic expressions using Pyvista.

    Keyword Arguments
    =================
        
        bg_color : int
            String or 3 element list for RGB. For example "white" or [1, 1, 1].
            Default to None. When None, pyvista's theme will be used.
        
        edge_color : string or 3 item list
            The solid color to give the edges when ``wireframe=True``.
            Either a string, RGB list, or hex color string. Default to "black".
        
        grid : boolean
            Show/Hide the gridlines. Default to False.
        
        style : str
            Visualization style of the mesh. Can be "surface", "wireframe" or
            "points". Note that "wireframe" only shows a wireframe of the outer
            geometry.
        
        tube_radius : float
            Tube radius for 3D lines. Default to 0.025.
        
        use_cm : boolean
            If True, apply a color map to the meshes/surface. If False, solid
            colors will be used instead. Default to True.
        
        wireframe : boolean
            Visualize the wireframe lines. It corresponds to pyvista's 
            ``show_edges`` keyword argument. Default to False.
    """

    colormaps = [
        "coolwarm", "kgy", "fire", "bmy", "bgy", "CET_D12", "rainbow", "kbc",
        "viridis", "plasma"
    ]

    def __new__(cls, *args, **kwargs):
        # Since Plot has its __new__ method, this will prevent infinite
        # recursion
        return object.__new__(cls)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # TODO: set here the backend used by Pyvista
        # https://docs.pyvista.org/user-guide/jupyter/index.html

        self._fig = pv.Plotter(
            window_size = [1024, 768] if not self.size else self.size,
            polygon_smoothing = True,
            line_smoothing = True
        )
        bg_color = kwargs.get("bg_color", pyvista_bg_color)
        if bg_color:
            self._fig.set_background(bg_color)
        self._fig.add_title("" if not self.title else self.title)
        self._fig.add_axes(
            xlabel=self.xlabel, ylabel=self.ylabel, zlabel=self.zlabel
        )
        if kwargs.get("grid", False):
            self._fig.show_grid(
                xlabel=self.xlabel, ylabel=self.ylabel, zlabel=self.zlabel
            )
    
    def self._init_cyclers(self):
        pass

    def _process_series(self, series):
        self._init_cyclers()

        for s in series:
            if s.is_3Dline:
                x, y, z = s.get_data()
                points = np.column_stack((x, y, z))
                line = self._polyline_from_points(points)
                line[str(s.var)] = s.discretized_var
                tube = line.tube(radius = self._kwargs.get("tube_radius", 0.025))

                args = dict(
                    color = next(self._iter_colorloop),
                    style = self._kwargs.get("style", "surface"),
                    show_edges = self._kwargs.get("wireframe", False),
                    edge_color = self._kwargs.get("edge_color", "black")
                )
                if self._use_cm:
                    args["scalars"] = str(s.var)
                    args["cmap"] = next(self._iter_colormaps)

                self._fig.add_mesh(tube, **args)

            elif s.is_3Dsurface:
                x, y, z = s.get_data()
                if s.is_parametric:
                    vertices, _indices = get_vertices_indices(x, y, z)
                    indices = np.ones((len(_indices), 4), dtype=np.uintc) * 3
                    indices[:, 1:] = _indices
                    surf = pv.PolyData(vertices, indices)
                else:
                    surf = pv.StructuredGrid(x, y, z)
                
                args = dict(
                    color = next(self._iter_colorloop),
                    style = self._kwargs.get("style", "surface"),
                    show_edges = self._kwargs.get("wireframe", False),
                    edge_color = self._kwargs.get("edge_color", "black"),
                )
                if self._use_cm:
                    args["scalars"] = z.reshape(-1)
                    args["cmap"] = next(self._iter_colormaps)
                self._fig.add_mesh(surf, **args)
            else:
                raise NotImplementedError(
                    "{} is not supported by {}\n".format(type(s), type(self).__name__) +
                    "Pyvista only supports 3D plots."
                )
    
    def _polyline_from_points(self, points):
        """Given an array of points, make a line set
        https://docs.pyvista.org/examples/00-load/create-spline.html#sphx-glr-examples-00-load-create-spline-py
        """
        poly = pv.PolyData()
        poly.points = points
        the_cell = np.arange(0, len(points), dtype=np.int_)
        the_cell = np.insert(the_cell, 0, len(points))
        poly.lines = the_cell
        return poly

    def show(self):
        self._process_series(self._series)
        self._fig.show()

PB = PyvistaBackend