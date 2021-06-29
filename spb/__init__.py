from spb.functions import (
    plot, plot_parametric, plot_contour,
    plot3d, plot3d_parametric_line, plot3d_parametric_surface,
    plot_implicit
)
from spb.plot_data import get_plot_data, smart_plot
from spb.interactive import iplot
from spb.vectors import vector_plot
from spb.complex.complex import complex_plot

# aliases
parametric_plot = plot_parametric
contour_plot = plot_contour
p3dpl = plot3d_parametric_line
p3dps = plot3d_parametric_surface
implicit_plot = plot_implicit
