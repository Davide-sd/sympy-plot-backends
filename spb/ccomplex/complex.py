from spb.plot_functions.complex import (
    plot_complex, plot_complex_list,
    plot_real_imag, plot_complex_vector, plot_riemann_sphere
)
import warnings

warnings.warn(
    "`spb.ccomplex.complex` is deprecated and will be removed in a future "
    "release. Please, use `spb.plot_functions` or `spb` directly "
    "(better option).",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "plot_complex", "plot_complex_list",
    "plot_real_imag", "plot_complex_vector", "plot_riemann_sphere"
]
