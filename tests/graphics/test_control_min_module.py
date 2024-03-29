import pytest
from pytest import warns, raises
from spb.series import LineOver1DRangeSeries, PoleZeroWithSympySeries
from spb.graphics.control import (
    step_response, impulse_response, ramp_response, pole_zero,
    root_locus, nyquist
)
from sympy.abc import s
from sympy.external import import_module
ct = import_module("control")


# verify that the minimal installation of the module is able to perform
# basic control-system plotting with SymPy.


@pytest.mark.skipif(ct is not None, reason="control is installed")
@pytest.mark.parametrize(
    "func, series_type", [
        (step_response, LineOver1DRangeSeries),
        (impulse_response, LineOver1DRangeSeries),
        (ramp_response, LineOver1DRangeSeries),
        (pole_zero, PoleZeroWithSympySeries)
    ]
)
def test_functions(func, series_type):
    # the plotting module informs the user that the evaluation will be done
    # with sympy (when ``control`` is not installed)
    G = (s + 1) / (s**2 + s + 1)
    
    with warns(
        UserWarning,
        match="``control=True`` was provided, but the ``control`` module"
    ):
        series = func(G, control=True)
    assert all(isinstance(s, series_type) for s in series)


@pytest.mark.skipif(ct is not None, reason="control is installed")
@pytest.mark.parametrize(
    "func", [root_locus, nyquist]
)
def test_errors(func):
    # the plotting module informs the user that the evaluation can't be done
    # because ``control`` is not installed
    G = (s + 1) / (s**2 + s + 1)

    raises(RuntimeError, lambda: func(G))
