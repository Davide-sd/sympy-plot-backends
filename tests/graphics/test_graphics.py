import numpy as np
import pytest
import spb
from spb import MB, PB, KB, BB
from spb.graphics import (
    graphics, line, line_polar,
    line_parametric_2d, line_parametric_3d,
)
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
)
from sympy import symbols, cos, sin, pi, exp
from sympy.external import import_module
from spb.interactive.panel import InteractivePlot as PanelInteractivePlot
from spb.interactive.ipywidgets import InteractivePlot as IPYInteractivePlot

pn = import_module("panel")


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings(
    "ignore:K3DBackend only works properly within Jupyter Notebook"
)
@pytest.mark.parametrize("backend", [MB, PB, KB, BB])
def test_graphics_single_series(backend):
    x = symbols("x")
    graphics_options = {"show": False, "backend": backend}

    g = graphics(line(cos(x)), **graphics_options)
    assert isinstance(g, backend)
    assert len(g.series) == 1
    assert isinstance(g.series[0], LineOver1DRangeSeries)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings(
    "ignore:K3DBackend only works properly within Jupyter Notebook"
)
@pytest.mark.parametrize("backend", [MB, PB, KB, BB])
def test_graphics_multiple_series(backend):
    x = symbols("x")
    graphics_options = {"show": False, "backend": backend}

    g = graphics(
        line(cos(x)),
        line_parametric_2d(cos(x), sin(x), (x, 0, pi)),
        line_parametric_3d(cos(x), sin(x), x, (x, 0, pi)),
        **graphics_options
    )
    assert isinstance(g, backend)
    assert len(g.series) == 3
    assert isinstance(g.series[0], LineOver1DRangeSeries)
    assert isinstance(g.series[1], Parametric2DLineSeries)
    assert isinstance(g.series[2], Parametric3DLineSeries)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings(
    "ignore:K3DBackend only works properly within Jupyter Notebook"
)
@pytest.mark.parametrize(
    "backend, imodule",
    [
        (MB, "panel"),
        (MB, "ipywidgets"),
        (PB, "panel"),
        (PB, "ipywidgets"),
        (BB, "panel"),
        (BB, "ipywidgets"),
        (KB, "panel"),
        (KB, "ipywidgets"),
    ],
)
def test_graphics_single_series_interactive(backend, imodule):
    x, u = symbols("x, u")
    graphics_options = {"show": False, "backend": backend, "imodule": imodule}

    g = graphics(line(cos(u * x), params={u: (1, 0, 2)}), **graphics_options)
    if imodule == "panel":
        g_type = PanelInteractivePlot
    else:
        g_type = IPYInteractivePlot
    assert isinstance(g, g_type)
    assert isinstance(g.backend, backend)
    assert len(g.backend.series) == 1


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings(
    "ignore:K3DBackend only works properly within Jupyter Notebook"
)
@pytest.mark.parametrize(
    "backend, imodule",
    [
        (MB, "panel"),
        (MB, "ipywidgets"),
        (PB, "panel"),
        (PB, "ipywidgets"),
        (BB, "panel"),
        (BB, "ipywidgets"),
        (KB, "panel"),
        (KB, "ipywidgets"),
    ],
)
def test_graphics_multiple_series_interactive(backend, imodule):
    # NOTE: these tests mixes 2D and 3D series. Here, they are not going to
    # raise errors because ``show=False``. All I care is to verify that the
    # correct classes are instantiated.

    x, u = symbols("x, u")
    graphics_options = {"show": False, "backend": backend, "imodule": imodule}

    g = graphics(
        line(cos(u * x), params={u: (1, 0, 2)}),
        line_parametric_2d(
            cos(u * x), sin(x), (x, 0, pi), params={u: (1, 0, 2)}
        ),
        line_parametric_3d(
            cos(u * x), sin(x), x, (x, 0, pi), params={u: (1, 0, 2)}
        ),
        **graphics_options
    )
    g_type = spb.interactive.panel.InteractivePlot if imodule == "panel" else spb.interactive.ipywidgets.InteractivePlot
    assert isinstance(g, g_type)
    assert isinstance(g.backend, backend)
    assert len(g.backend.series) == 3
    assert isinstance(g.backend.series[0], LineOver1DRangeSeries)
    assert isinstance(g.backend.series[1], Parametric2DLineSeries)
    assert isinstance(g.backend.series[2], Parametric3DLineSeries)


def test_graphics_polar_axis():
    # verify that polar_axis=True set the appropriate transformation on
    # data series

    t = symbols("t")
    g1 = graphics(
        line_polar(exp(sin(t)) - 2 * cos(4 * t), (t, 0, 2 * pi)),
        show=False,
        polar_axis=False,
    )
    d1 = g1[0].get_data()
    g2 = graphics(
        line_polar(exp(sin(t)) - 2 * cos(4 * t), (t, 0, 2 * pi)),
        show=False,
        polar_axis=True,
    )
    d2 = g2[0].get_data()
    assert not np.allclose(d1, d2)
