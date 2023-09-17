from spb.plot_functions.vectors import plot_vector
import warnings

warnings.warn(
    "`spb.vectors` is deprecated and will be removed in a future release. "
    "Please, use `spb.plot_functions` or `spb` directly (better option).",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["plot_vector"]
