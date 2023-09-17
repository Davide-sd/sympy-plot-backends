from spb.plot_functions.control import (
    pole_zero_plot, plot_pole_zero,
    step_response_plot, plot_step_response,
    impulse_response_plot, plot_impulse_response,
    ramp_response_plot, plot_ramp_response,
    bode_magnitude_plot, plot_bode_magnitude,
    bode_phase_plot, plot_bode_phase,
    bode_plot, plot_bode,
    nyquist_plot, plot_nyquist,
    nichols_plot, plot_nichols
)
import warnings

warnings.warn(
    "`spb.control` is deprecated and will be removed in a future release. "
    "Please, use `spb.plot_functions.control` or "
    "`spb` directly (better option).",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'pole_zero_plot', 'plot_pole_zero',
    'step_response_plot', 'plot_step_response',
    'impulse_response_plot', 'plot_impulse_response',
    'ramp_response_plot', 'plot_ramp_response',
    'bode_magnitude_plot', 'plot_bode_magnitude',
    'bode_phase_plot', 'plot_bode_phase',
    'bode_plot', 'plot_bode',
    'nyquist_plot', 'plot_nyquist',
    'nichols_plot', 'plot_nichols'
]