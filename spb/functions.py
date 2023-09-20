from spb.plot_functions.functions_2d import (
    _set_labels,
    _create_generic_data_series,
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
import warnings

warnings.warn(
    "`spb.functions` is deprecated and will be removed in a future release. "
    "Please, use `spb.plot_functions_2d` or `spb.plot_functions_3d` or "
    "`spb` directly (better option).",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "_set_labels",
    "_create_generic_data_series",
    "plot",
    "plot_parametric",
    "plot_parametric_region",
    "plot_contour",
    "plot_implicit",
    "plot_polar",
    "plot_geometry",
    "plot_list",
    "plot_piecewise",
    "plot3d",
    "plot3d_parametric_line",
    "plot3d_parametric_surface",
    "plot3d_implicit",
    "plot3d_spherical",
    "plot3d_revolution",
    "plot3d_list"
]
