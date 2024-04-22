from spb.series import BaseSeries, HVLineSeries
import pytest
from pytest import raises
import matplotlib
import numpy as np
from spb import (
    PB, MB, KB, BB, plot, plot3d, prange, plot_vector, plot_contour,
    plot_complex, plot_parametric
)
from sympy import sin, cos, pi, exp, symbols, I
from sympy.external import import_module

pn = import_module("panel")
ipy = import_module("ipywidgets")

KB.skip_notebook_check = True


class UnsupportedSeries(BaseSeries):
    pass


def test_unsupported_series():
    # verify that an error is raised when an unsupported series is given in
    series = [UnsupportedSeries()]
    raises(
        NotImplementedError,
        lambda: MB(*series).draw())
    if pn is not None:
        raises(
            NotImplementedError,
            lambda: PB(*series).draw())
        raises(
            NotImplementedError,
            lambda: BB(*series).draw())
        raises(
            NotImplementedError,
            lambda: KB(*series).draw())


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_common_keywords():
    # TODO: here I didn't test axis_center, autoscale, margin
    kw = dict(
        title="a",
        xlabel="x",
        ylabel="y",
        zlabel="z",
        aspect="equal",
        grid=False,
        xscale="log",
        yscale="log",
        zscale="log",
        legend=True,
        xlim=(-1, 1),
        ylim=(-2, 2),
        zlim=(-3, 3),
        size=(5, 10),
    )
    p = BB(**kw)
    assert p.title == "a"
    assert p.xlabel == "x"
    assert p.ylabel == "y"
    assert p.zlabel == "z"
    assert p.aspect == "equal"
    assert p.grid is False
    assert p.xscale == "log"
    assert p.yscale == "log"
    assert p.zscale == "log"
    assert p.legend is True
    assert p.xlim == (-1, 1)
    assert p.ylim == (-2, 2)
    assert p.zlim == (-3, 3)
    assert p.size == (5, 10)


@pytest.mark.parametrize(
    "Backend", [MB, PB, KB, BB]
)
def test_colorloop_colormaps(Backend):
    # verify that backends exposes important class attributes enabling
    # automatic coloring

    assert hasattr(Backend, "colorloop")
    assert isinstance(Backend.colorloop, (list, tuple))
    assert hasattr(Backend, "colormaps")
    assert isinstance(Backend.colormaps, (list, tuple))


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_plot_sum():
    x, y = symbols("x, y")

    options = dict(adaptive=False, n=5, show=False)

    # the choice of the backend dictates the keyword arguments
    # inside rendering_kw
    p1 = plot(
        sin(x), backend=PB, rendering_kw=dict(line_color='black'),
        xlabel="x1", ylabel="y1", **options
    )
    p2 = plot(
        cos(x), backend=PB, rendering_kw=dict(line_dash='dash'),
        xlabel="x2", ylabel="y2", **options
    )
    p3 = plot(
        sin(x) * cos(x), backend=PB,
        rendering_kw=dict(line_dash='dot'), **options
    )
    p4 = p1 + p2 + p3
    assert isinstance(p4, PB)
    assert len(p4.series) == 3
    assert p4.series[0].expr == sin(x)
    assert p4.fig.data[0]["line"]["color"] == "black"
    colors = set()
    for l in p4.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 3
    assert p4.series[1].expr == cos(x)
    assert p4.fig.data[1]["line"]["dash"] == "dash"
    assert p4.series[2].expr == sin(x) * cos(x)
    assert p4.fig.data[2]["line"]["dash"] == "dot"
    # two or more series in the result: automatic legend turned on
    assert p4.legend is True
    # the resulting plot uses the attributes of the first plot in the sum
    assert p4.xlabel == "x1" and p4.ylabel == "y1"
    p4 = p2 + p1 + p3
    assert p4.xlabel == "x2" and p4.ylabel == "y2"

    # plots can be added together with sum()
    p5 = sum([p1, p2, p3])
    assert isinstance(p5, PB)
    assert len(p5.series) == 3

    # summing different types of plots: the result is consistent with the
    # original visualization. In particular, if no `rendering_kw` is given
    # to `p2` then the backend will use automatic coloring to differentiate
    # the series.
    hex2rgb = lambda h: tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    p1 = plot_vector(
        [-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        backend=MB, scalar=True, **options
    )
    p2 = plot(sin(x), (x, -3, 3), backend=MB, **options)
    p3 = p1 + p2
    assert isinstance(p3, MB)
    assert len(p3.series) == 3
    assert len(p3.fig.axes[0].collections) > 1
    assert isinstance(p3.fig.axes[0].collections[-1], matplotlib.quiver.Quiver)
    quiver_col = p3.fig.axes[0].collections[-1].get_facecolors().flatten()[:-1]
    first_col = np.array(p3.colorloop[0])
    if quiver_col.dtype == first_col.dtype:
        assert np.allclose(quiver_col, first_col)
    else:
        assert np.allclose(quiver_col, hex2rgb(str(first_col)[1:]))
    line_col = np.array(p3.fig.axes[0].lines[0].get_color())
    second_col = np.array(p3.colorloop[1])
    if line_col.dtype == second_col.dtype == "<U7":
        assert str(line_col) == str(second_col)
    else:
        assert np.allclose(line_col, hex2rgb(str(second_col)[1:]))

    # summing plots with different backends: the first backend will be used in
    # the result
    p1 = plot(sin(x), backend=MB, **options)
    p2 = plot(cos(x), backend=PB, **options)
    p3 = p1 + p2
    assert isinstance(p3, MB)

    # summing plots with different backends: fail when backend-specific
    # keyword arguments are used.
    # NOTE: the output plot is of type MB
    p1 = plot(
        sin(x), backend=MB, rendering_kw=dict(linestyle=":"), **options
    )
    p2 = plot(
        cos(x), backend=PB, rendering_kw=dict(line_dash="dash"), **options
    )
    raises(AttributeError, lambda: (p1 + p2).draw())

    # verify that summing up bokeh plots doesn't raise errors
    p1 = plot(sin(x), (x, -pi, pi), backend=BB, **options)
    p2 = plot(cos(x), (x, -pi, pi), backend=BB, **options)
    p3 = p1 + p2

    # verify that summing up K3D plots doesn't raise errors
    p1 = plot3d(
        sin(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), backend=KB,
        **options
    )
    p2 = plot3d(
        cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), backend=KB,
        **options
    )
    p3 = p1 + p2


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_xaxis_inverted():
    # verify that no errors are raised when parametric ranges are used

    x, a, b, c, n = symbols("x, a, b, c, n")
    p = plot(
        (cos(a * x + b) * exp(-c * x), "oscillator"),
        (exp(-c * x), "upper limit", {"linestyle": ":"}),
        (-exp(-c * x), "lower limit", {"linestyle": ":"}),
        prange(x, 0, n * pi),
        params={
            a: (1, 0, 10),      # frequency
            b: (0, 0, 2 * pi),  # phase
            c: (0.25, 0, 1),    # damping
            n: (2, 0, 4)        # multiple of pi
        },
        ylim=(-1.25, 1.25), backend=MB, use_latex=False, show=False, n=10
    )
    p.backend.draw()


def test_number_of_renderers():
    # verify that once `extend` or `append` is executed, the number of
    # renderers will be equal to the new number of series

    x = symbols("x")
    hor = HVLineSeries(0, True, show_in_legend=False)
    ver1 = HVLineSeries(0, False, show_in_legend=False)
    ver2 = HVLineSeries(1, False, show_in_legend=False)

    def do_test(B):
        p = plot(sin(x), (x, -5, 5), backend=B, show=False, n=5)
        assert len(p.series) == len(p.renderers) == 1

        p.extend([hor, ver1])
        assert len(p.series) == len(p.renderers) == 3

        p.append(ver2)
        assert len(p.series) == len(p.renderers) == 4

    do_test(MB)
    if pn is not None:
        do_test(PB)
        do_test(BB)


def test_axis_scales():
    # by default, axis scales should be set to None: this allows users to
    # extend plot capabilities to create plots with categoricals axis.
    # See: https://github.com/Davide-sd/sympy-plot-backends/issues/29
    x = symbols("x")
    p = plot(sin(x), backend=MB, show=False, n=5)
    assert all(t is None for t in [p.xscale, p.yscale, p.zscale])

    p = plot(sin(x), backend=MB, show=False, n=5, xscale="linear")
    assert p.xscale == "linear"
    assert all(t is None for t in [p.yscale, p.zscale])

    p = plot(sin(x), backend=MB, show=False, n=5, yscale="linear")
    assert p.yscale == "linear"
    assert all(t is None for t in [p.xscale, p.zscale])

    p = plot(sin(x), backend=MB, show=False, n=5, zscale="linear")
    assert p.zscale == "linear"
    assert all(t is None for t in [p.xscale, p.yscale])


@pytest.mark.parametrize(
    "backend", [MB, PB, BB]
)
def test_update_event_numeric_ranges(backend):
    # verify that the code responsible for updating the ranges works as
    # expected.

    if pn is None:
        # bokeh and plotly are not installed
        return

    x, y = symbols("x, y")
    p = plot(sin(x), (x, -pi, pi), show=False, n=10, backend=backend)
    assert p[0].ranges == [(x, -pi, pi)]
    p._update_series_ranges((-10, 10))
    assert p[0].ranges == [(x, -10, 10)]

    p = plot_contour(
        cos(x**2 + y**2) * exp(-(x**2 + y**2) / 5), (x, -pi, pi), (y, -pi, pi),
        show=False, n=10, backend=backend
    )
    assert p[0].ranges == [(x, -pi, pi), (y, -pi, pi)]
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[0].ranges == [(x, -10, 10), (y, -5, 6)]

    p = plot_vector(
        [-sin(y), cos(x)], (x, -pi, pi), (y, -pi, pi),
        backend=backend, scalar=True, n=10, show=False
    )
    assert p[0].ranges == [(x, -pi, pi), (y, -pi, pi)]
    assert p[1].ranges == [(x, -pi, pi), (y, -pi, pi)]
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[1].ranges == [(x, -10, 10), (y, -5, 6)]
    assert p[0].ranges == [(x, -10, 10), (y, -5, 6)]

    p = plot_complex(sin(x), (x, -2-2j, 2+2j),
        show=False, n=10, backend=backend)
    assert p[0].ranges[0][0] == x
    assert (p[0].ranges[0][1] - (-2 - 2*I)).nsimplify() == 0
    assert (p[0].ranges[0][2] - (2 + 2*I)).nsimplify() == 0
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[0].ranges[0][0] == x
    assert (p[0].ranges[0][1] - (-10 - 5*I)).nsimplify() == 0
    assert (p[0].ranges[0][2] - (10 + 6*I)).nsimplify() == 0

    # parametric plot must not update the range
    p = plot_parametric(cos(x), sin(x), (x, 0, 2*pi),
        show=False, backend=backend, n=10)
    assert p[0].ranges == [(x, 0, 2*pi)]
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[0].ranges == [(x, 0, 2*pi)]


@pytest.mark.parametrize(
    "backend", [MB, PB, BB]
)
def test_update_event_parametric_ranges(backend):
    # verify that the code responsible for updating the ranges works as
    # expected.

    if pn is None:
        # bokeh and plotly are not installed
        return

    x, y, z = symbols("x, y, z")
    ip = plot(sin(x), prange(x, -y*pi, pi), show=False, n=10, backend=backend,
        params={y: (1, 0, 2)})
    p = ip._backend
    assert p[0].ranges == [(x, -y*pi, pi)]
    p._update_series_ranges((-10, 10))
    assert p[0].ranges == [(x, -y*pi, pi)]

    ip = plot_contour(
        cos(x**2 + y**2) * exp(-(x**2 + y**2) / 5),
        prange(x, -z*pi, pi), (y, -pi, pi),
        show=False, n=10, backend=backend, params={z: (1, 0, 2)}
    )
    p = ip._backend
    assert p[0].ranges == [(x, -z*pi, pi), (y, -pi, pi)]
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[0].ranges == [(x, -z*pi, pi), (y, -5, 6)]

    ip = plot_vector(
        [-sin(y), cos(x)], prange(x, -z*pi, pi), (y, -pi, pi),
        backend=backend, scalar=True, n=10, show=False, params={z: (1, 0, 2)}
    )
    p = ip._backend
    assert p[0].ranges == [(x, -z*pi, pi), (y, -pi, pi)]
    assert p[1].ranges == [(x, -z*pi, pi), (y, -pi, pi)]
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[1].ranges == [(x, -z*pi, pi), (y, -5, 6)]
    assert p[0].ranges == [(x, -z*pi, pi), (y, -5, 6)]

    ip = plot_complex(sin(x), prange(x, -y-2j, 2+2j),
        show=False, n=10, backend=backend, params={y: (1, 0, 2)})
    p = ip._backend
    assert p[0].ranges[0][0] == x
    assert (p[0].ranges[0][1] - (-y - 2*I)).nsimplify() == 0
    assert (p[0].ranges[0][2] - (2 + 2*I)).nsimplify() == 0
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[0].ranges[0][0] == x
    assert (p[0].ranges[0][1] - (-y - 2*I)).nsimplify() == 0
    assert (p[0].ranges[0][2] - (2 + 2*I)).nsimplify() == 0

    # parametric plot must not update the range
    ip = plot_parametric(cos(x), sin(x), prange(x, y, 2*pi),
        show=False, backend=PB, n=10, params={y: (1, 0, 2)})
    p = ip._backend
    assert p[0].ranges == [(x, y, 2*pi)]
    p._update_series_ranges((-10, 10), (-5, 6))
    assert p[0].ranges == [(x, y, 2*pi)]
