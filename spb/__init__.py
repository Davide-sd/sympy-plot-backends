from spb._version import __version__
from spb.functions import (
    plot,
    plot_parametric,
    plot_contour,
    plot3d,
    plot3d_parametric_line,
    plot3d_parametric_surface,
    plot_implicit,
    plot_polar,
    plot_geometry,
)
from spb.plot_data import get_plot_data, smart_plot
from spb.vectors import plot_vector
from spb.ccomplex.complex import (
    plot_complex, plot_complex_list,
    plot_real_imag, plot_complex_vector
)

# from spb.interactive import iplot
# from spb.backends.plotgrid import plotgrid

# aliases
parametric_plot = plot_parametric
contour_plot = plot_contour
p3d = plot3d
p3dpl = plot3d_parametric_line
p3dps = plot3d_parametric_surface
implicit_plot = plot_implicit
polar_plot = plot_polar
geometry_plot = plot_geometry
complex_plot = plot_complex
vector_plot = plot_vector
