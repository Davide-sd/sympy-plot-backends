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
    plot_list,
    plot_piecewise
)
from spb.plot_data import get_plot_data, smart_plot
from spb.vectors import plot_vector
from spb.ccomplex.complex import (
    plot_complex, plot_complex_list,
    plot_real_imag, plot_complex_vector
)

from spb.plotgrid import plotgrid
# NOTE: it would be nice to have `iplot` readily available, however loading
# `panel` is a slow operation.
# from spb.interactive import iplot
