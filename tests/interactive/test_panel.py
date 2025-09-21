import pytest
from pytest import raises
pn = pytest.importorskip("panel")
from spb import (
    BB, PB, MB, plot, graphics, line, line_parametric_2d,
    domain_coloring
)
from spb.interactive.panel import (
    DynamicParam, InteractivePlot, create_widgets
)
from spb.interactive.bootstrap_spb import SymPyBootstrapTemplate
from spb.utils import prange
from sympy import (
    Integer, Float, Rational, pi, symbols, sin, cos, exp
)
import numpy as np
import param
import bokeh


@pytest.mark.parametrize(
    "par, expected_widget_type", [
        (
            param.Number(default=0, bounds=(0, 2), label="test"),
            pn.widgets.FloatSlider
        ),
        (
            param.Integer(default=0, bounds=(0, 2), label="test"),
            pn.widgets.IntSlider
        ),
        (
            param.Boolean(True),
            pn.widgets.Checkbox
        ),
        (
            param.ObjectSelector(
                default=5, objects=[1, 2, 3, 4, 5], label="test5"
            ),
            pn.widgets.select.Select
        )
    ]
)
def test_DynamicParam(par, expected_widget_type):
    # verify the proper conversion from a param's Parameter to a
    # panel's widget
    d = DynamicParam(par)
    tmp_panel = pn.Param(d)
    widget = tmp_panel.widget("dyn_param_0")
    assert isinstance(widget, expected_widget_type)


def test_iplot(panel_options):
    # verify that the correct widgets are created, with the appropriate labels

    bm = bokeh.models
    a, b, c, d, e, f, g, h, x = symbols("a, b, c, d, e, f, g, h, x")

    t = plot(
        c * cos(a * x) * exp(-abs(x) / b) + d * (x - e) * f + g*h,
        (x, -5, 5),
        params={
            a: (2, 1, 3, 5),
            b: (3, 1e-03, 4e03, 10, None, "label", "log"),
            c: param.Number(0.15, softbounds=(0, 1), label="test", step=0.025),
            # TODO: if I remove the following label, the tests are going to
            # fail: it would use the label "test5"... How is it possible?
            d: param.Integer(1, softbounds=(0, 10), label="d"),
            e: param.Integer(1, softbounds=(0, None)),
            f: param.Boolean(default=True),
            g: param.ObjectSelector(default=2, objects=[1, 2, 3, 4]),
            h: pn.widgets.FloatSlider(
                value=5, start=0, end=10, name="float slider")
        },
        layout="tb",
        ncols=2,
        use_latex=False,
        **panel_options
    )

    # c1 wraps the controls, c2 wraps the plot
    gridbox, html = t.show().get_root().children
    assert isinstance(gridbox.children[0][0], bm.Slider)
    assert isinstance(gridbox.children[1][0].children[1], bm.Slider)
    assert isinstance(gridbox.children[2][0], bm.Slider)
    assert isinstance(gridbox.children[3][0], bm.Slider)
    assert isinstance(gridbox.children[4][0], bm.Spinner)
    assert isinstance(gridbox.children[5][0], bm.Checkbox)
    assert isinstance(gridbox.children[6][0], bm.Select)
    assert isinstance(gridbox.children[7][0], bm.Slider)
    assert gridbox.children[0][0].title == "a"
    assert gridbox.children[1][0].children[0].text[:5] == "label"
    assert gridbox.children[2][0].title == "test"
    assert gridbox.children[3][0].title == "d"
    assert gridbox.children[7][0].title == "float slider"
    # verify it's an Nx2 grid of widgets
    assert gridbox.children[0][1:3] == (0, 0)
    assert gridbox.children[1][1:3] == (0, 1)
    assert gridbox.children[2][1:3] == (1, 0)
    assert gridbox.children[3][1:3] == (1, 1)
    assert gridbox.children[4][1:3] == (2, 0)
    assert gridbox.children[5][1:3] == (2, 1)
    assert gridbox.children[6][1:3] == (3, 0)
    assert gridbox.children[7][1:3] == (3, 1)

    # verify use_latex works properly
    t = plot(
        (a + b + c) * cos(x),
        (x, -5, 5),
        params={
            a: (2, 1, 3, 5),
            b: (3, 2, 4000, 10),
            c: param.Number(0.15, softbounds=(0, 1), label="test", step=0.025),
        },
        layout="tb",
        ncols=3,
        use_latex=True,
        **panel_options
    )

    # c1 wraps the controls, c2 wraps the plot
    gridbox, html = t.show().get_root().children
    assert gridbox.children[0][0].title == "$$a$$"
    assert gridbox.children[1][0].title == "$$b$$"
    assert gridbox.children[2][0].title == "test"
    # verify it's an 1x3 grid of widgets
    assert gridbox.children[0][1:3] == (0, 0)
    assert gridbox.children[1][1:3] == (0, 1)
    assert gridbox.children[2][1:3] == (0, 2)


def test_template(panel_options):
    # verify that user can provide custom templates for servable applications
    # (the ones that launches on a new browser window instead of being shown
    # on Jupyter Notebook).
    # NOTE: I'm not aware of any way to test an application once they have been
    # served, hence I'm only going to test if the user can change the default
    # template.

    x, y = symbols("x, y")

    # default template
    p = plot(
        cos(x * y), (x, -5, 5),
        params={y: (1, 0, 2)},
        servable=True,
        **panel_options
    )
    t = p._create_template()
    assert isinstance(t, SymPyBootstrapTemplate)

    # default template with customized settings
    p = plot(
        cos(x * y), (x, -5, 5),
        params={y: (1, 0, 2)},
        servable=True,
        template={
            "title": "Test",
            "full_width": False,
            "sidebar_width": "50%",
            "header_no_panning": False,
            "sidebar_location": "tb",
        },
        **panel_options
    )
    t = p._create_template()
    assert isinstance(t, SymPyBootstrapTemplate)

    # a different template
    p = plot(
        cos(x * y), (x, -5, 5),
        params={y: (1, 0, 2)},
        servable=True,
        template=pn.template.VanillaTemplate,
        **panel_options
    )
    t = p._create_template()
    assert isinstance(t, pn.template.VanillaTemplate)

    # an instance of a different template
    temp = pn.template.MaterialTemplate
    p = plot(
        cos(x * y), (x, -5, 5),
        params={y: (1, 0, 2)},
        servable=True,
        template=temp(),
        **panel_options
    )
    t = p._create_template()
    assert isinstance(t, pn.template.MaterialTemplate)

    # something not supported
    p = plot(
        cos(x * y), (x, -5, 5),
        params={y: (1, 0, 2)},
        servable=True,
        template="str",
        **panel_options
    )
    raises(TypeError, lambda: p._create_template())


def test_create_widgets():
    x, y, z = symbols("x:z")

    w = create_widgets(
        {
            x: (2, 0, 4),
            y: (200, 1, 1000, 10, None, "y", "log"),
            z: param.Integer(3, softbounds=(3, 10), label="n"),
        },
        use_latex=True,
    )

    assert isinstance(w, dict)
    assert len(w) == 3
    assert isinstance(w[x], pn.widgets.FloatSlider)
    assert isinstance(w[y], pn.widgets.DiscreteSlider)
    assert isinstance(w[z], pn.widgets.IntSlider)
    assert w[x].name == "$$x$$"
    assert w[y].name == "y"
    assert w[z].name == "n"

    formatter = bokeh.models.formatters.PrintfTickFormatter(format="%.4f")
    w = create_widgets(
        {
            x: (2, 0, 4, formatter),
            y: (200, 1, 1000, 10, "%.4f", "y", "log"),
            z: param.Integer(3, softbounds=(3, 10), label="n"),
        },
        use_latex=False,
    )

    assert isinstance(w, dict)
    assert len(w) == 3
    assert isinstance(w[x], pn.widgets.FloatSlider)
    assert isinstance(w[y], pn.widgets.DiscreteSlider)
    assert isinstance(w[z], pn.widgets.IntSlider)
    assert w[x].name == "x"
    assert w[y].name == "y"
    assert w[z].name == "n"

    assert isinstance(w[x].format, bokeh.models.formatters.PrintfTickFormatter)
    assert w[y].formatter == "%.4f"
    assert w[z].format is None


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_params_multi_value_widgets_1():
    a, b, c, x = symbols("a:c x")
    p = plot(
        cos(c * x), prange(x, a, b), n=10,
        params={
            (a, b): pn.widgets.RangeSlider(
                value=(-2, 2), start=-5, end=5, step=0.1),
            c: (1, 0, 5)
        }, imodule="panel", show=False, backend=MB)
    fig = p.fig
    d1 = p.backend[0].get_data()
    # verify that no errors are raise when an update event is triggered
    p._update((-3, 3), 2)
    d2 = p.backend[0].get_data()
    assert not np.allclose(d1, d2)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_params_multi_value_widgets_2():
    a, b, c, d, x = symbols("a:d x")
    p = graphics(
        line(cos(x), range_x=(x, -5, 5), n=10),
        line(cos(c * x), range_x=(x, a, b), n=10,
            params={
                (a, b): pn.widgets.RangeSlider(
                    value=(-2, 2), start=-5, end=5, step=0.1),
                c: (1, 0, 5)
            }),
        line_parametric_2d(cos(d*x), sin(d*x), range_p=(x, -4, 4),
            n=10,
            params={d: (2, 0, 6)}),
        imodule="panel", show=False, backend=MB)
    fig = p.fig
    d1 = p.backend[0].get_data()
    d2 = p.backend[1].get_data()
    d3 = p.backend[2].get_data()
    # verify that no errors are raise when an update event is triggered
    p._update((-3, 3), 2, 3)
    d4 = p.backend[0].get_data()
    d5 = p.backend[1].get_data()
    d6 = p.backend[2].get_data()
    assert np.allclose(d1, d4)
    assert not np.allclose(d2, d5)
    assert not np.allclose(d3, d6)


@pytest.mark.parametrize("backend, interactive_series", [
    (MB, True),
    (MB, False),
    (BB, True),
    (BB, False)
])
def test_domain_coloring_series_ui_controls(backend, interactive_series):
    # verify that UI controls related to ComplexDomainColoringSeries
    # are added to the interactive application

    x, u = symbols("x, u")
    s1 = domain_coloring(
            sin(x), (x, -2-2j, 2+2j), n=10)
    s2 = domain_coloring(
            sin(u*x), (x, -2-2j, 2+2j), params={u: (1, 0, 2)}, n=10)

    p = graphics(
        s2 if interactive_series else s1,
        backend=backend,
        grid=False,
        imodule="panel",
        layout="sbl",
        ncols=1,
        show=False,
        app=True if not interactive_series else None
    )
    assert isinstance(p, InteractivePlot)
    fig = p.fig
    s = p.backend[0]
    assert s.coloring == "a"
    _, _, _, _, img1, _ = s.get_data()

    # verify that no errors are raise when an update event is triggered
    new_data = [1, 10, 10, "b", 20, 0.75, 0]
    if not interactive_series:
        new_data = [10, 10, "b", 20, 0.75, 0]
    p._update(*new_data)
    _, _, _, _, img2, _ = s.get_data()
    assert not np.allclose(img1, img2)

    if interactive_series:
        new_data[1] = 20
    else:
        new_data[0] = 20
    p._update(*new_data)
    _, _, _, _, img3, _ = s.get_data()
    assert img2.shape != img3.shape
