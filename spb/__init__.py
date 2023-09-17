from spb._version import __version__
from spb.plot_functions.functions_2d import (
    plot,
    plot_parametric,
    plot_parametric_region,
    plot_contour,
    plot_implicit,
    plot_polar,
    plot_geometry,
    plot_list,
    plot_piecewise
)
from spb.plot_functions.functions_3d import (
    plot3d,
    plot3d_parametric_line,
    plot3d_parametric_surface,
    plot3d_implicit,
    plot3d_spherical,
    plot3d_revolution,
    plot3d_list
)
from spb.plot_functions.vectors import plot_vector
from spb.plot_functions.complex_analysis import (
    plot_complex, plot_complex_list,
    plot_real_imag, plot_complex_vector, plot_riemann_sphere
)
from spb.plot_functions.control import (
    plot_pole_zero, plot_step_response, plot_impulse_response,
    plot_ramp_response, plot_bode_magnitude, plot_bode_phase, plot_bode,
    plot_nyquist, plot_nichols
)
from spb.utils import prange

from spb.plotgrid import plotgrid, PlotGrid

from spb.backends.matplotlib import MB
from spb.backends.bokeh import BB
from spb.backends.plotly import PB
from spb.backends.k3d import KB
from spb.backends.mayavi import MAB

from spb.graphics.graphics import graphics
from spb.graphics.functions_2d import (
    line, line_parametric_2d, line_polar, contour, implicit_2d, list_2d,
    geometry, hline, vline
)
from spb.graphics.functions_3d import (
    surface, surface_parametric, surface_revolution, surface_spherical,
    line_parametric_3d, list_3d, implicit_3d, wireframes, plane
)
from spb.graphics.vectors import (
    vector_field_2d, vector_field_3d
)
from spb.graphics.complex_analysis import (
    complex_points, line_abs_arg, line_abs_arg_colored, line_real_imag,
    surface_abs_arg, surface_real_imag, domain_coloring, analytic_landscape,
    riemann_sphere_2d, riemann_sphere_3d, complex_vector_field,
    contour_real_imag, contour_abs_arg
)
from spb.graphics.control import (
    control_axis, pole_zero, step_response, impulse_response, ramp_response,
    bode_magnitude, bode_phase, nyquist, nichols
)

__all__ = [
    "__version__", "plot", "plot_parametric", "plot_parametric_region",
    "plot_contour", "plot3d", "plot3d_parametric_line",
    "plot3d_parametric_surface", "plot3d_implicit", "plot3d_spherical",
    "plot3d_revolution", "plot_implicit", "plot_polar", "plot_geometry",
    "plot_list", "plot_piecewise", "plot_vector", "plot_complex",
    "plot_complex_list", "plot_real_imag", "plot_complex_vector", "plotgrid",
    "MB", "BB", "PB", "KB", "MAB", "plot3d_list", "prange",
    "plot_riemann_sphere", "PlotGrid",
    "plot_pole_zero", "plot_step_response", "plot_impulse_response",
    "plot_ramp_response", "plot_bode_magnitude", "plot_bode_phase",
    "plot_bode", "plot_nyquist", "plot_nichols",
    "graphics", "line", "line_parametric_2d", "line_polar", "implicit_2d",
    "list_2d", "geometry", "contour", "surface", "surface_parametric",
    "surface_spherical", "surface_revolution", "line_parametric_3d",
    "implicit_3d", "list_3d", "wireframes", "vector_field_2d",
    "vector_field_3d", "complex_points", "line_abs_arg_colored",
    "line_abs_arg", "line_real_imag", "surface_abs_arg", "surface_real_imag",
    "domain_coloring", "analytic_landscape", "riemann_sphere_2d",
    "riemann_sphere_3d", "complex_vector_field",
    "contour_real_imag", "contour_abs_arg", "plane",
    "hline", "vline",
    "pole_zero", "control_axis", "step_response", "impulse_response",
    "ramp_response", "bode_magnitude", "bode_phase", "nyquist", "nichols"
]
