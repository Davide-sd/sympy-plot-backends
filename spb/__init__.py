from spb.functions import (
    plot, plot_parametric, plot_contour,
    plot3d, plot3d_parametric_line, plot3d_parametric_surface,
    plot_implicit, polar_plot, geometry_plot
)
from spb.plot_data import get_plot_data, smart_plot
# from spb.interactive import iplot
from spb.vectors import vector_plot
from spb.complex.complex import complex_plot
# from spb.backends.plotgrid import plotgrid

# aliases
parametric_plot = plot_parametric
contour_plot = plot_contour
p3d = plot3d
p3dpl = plot3d_parametric_line
p3dps = plot3d_parametric_surface
implicit_plot = plot_implicit
plot_polar = polar_plot
plot_geometry = geometry_plot
plot_complex = complex_plot

__all__ = [
    "plot", "plot_parametric", "plot_contour",
    "plot3d", "plot3d_parametric_line", "plot3d_parametric_surface",
    "plot_implicit", "polar_plot", "geometry_plot",
    "get_plot_data", "smart_plot",
    "vector_plot",
    "complex_plot",
    "parametric_plot", "contour_plot", "p3dpl", "p3dps", "p3d", 
    "implicit_plot", "plot_polar", "plot_geometry", "plot_complex"
]
