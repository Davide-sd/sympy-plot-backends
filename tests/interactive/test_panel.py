import pytest
from pytest import raises
pn = pytest.importorskip("panel")
from spb import BB, PB, MB, plot
from spb.interactive.panel import (
    DynamicParam, MyList, InteractivePlot, create_widgets
)
from spb.interactive.bootstrap_spb import SymPyBootstrapTemplate
from sympy import (
    Integer, Float, Rational, pi, symbols, sin, cos
)
import numpy as np
import param
import bokeh


def test_DynamicParam():
    a, b, c, d, e, f = symbols("a, b, c, d, e, f")

    # test _tuple_to_dict
    t = DynamicParam(
        params={
            a: (1, 0, 5),
            b: (2, 1.5, 4.5, 20),
            c: (3, 2, 5, 30, None, "test1"),
            d: (1, 1, 10, 10, None, "test2", "log"),
        },
        use_latex=False,
    )
    p1 = getattr(t.param, "dyn_param_0")
    p2 = getattr(t.param, "dyn_param_1")
    p3 = getattr(t.param, "dyn_param_2")
    p4 = getattr(t.param, "dyn_param_3")

    def test_number(p, d, sb, l, st):
        assert isinstance(p, param.Number)
        assert p.default == d
        assert p.softbounds == sb
        assert p.label == l
        assert p.step == st

    def test_log_slider(p, d, sb, n, l):
        assert isinstance(p, MyList)
        assert p.default == 1
        assert p.objects[0] == sb[0]
        assert p.objects[-1] == sb[1]
        assert len(p.objects) == 10
        assert p.label == l

    test_number(p1, 1, (0, 5), "a", 0.125)
    test_number(p2, 2, (1.5, 4.5), "b", 0.15)
    test_number(p3, 3, (2, 5), "test1", 0.1)
    test_log_slider(p4, 1, (1, 10), 10, "test2")

    # all formatters should be None
    assert isinstance(t.formatters, dict)
    assert len(t.formatters) == 4
    assert all(e is None for e in t.formatters.values())

    # test use_latex
    formatter = bokeh.models.formatters.PrintfTickFormatter(format="%.4f")
    t = DynamicParam(
        params={
            a: (1, 0, 5),
            b: (2, 1.5, 4.5, 20),
            c: (3, 2, 5, 30, formatter, "test1"),
            d: (1, 1, 10, 10, None, "test2", "log"),
        },
        use_latex=True,
    )
    p1 = getattr(t.param, "dyn_param_0")
    p2 = getattr(t.param, "dyn_param_1")
    p3 = getattr(t.param, "dyn_param_2")
    p4 = getattr(t.param, "dyn_param_3")

    test_number(p1, 1, (0, 5), "$$a$$", 0.125)
    test_number(p2, 2, (1.5, 4.5), "$$b$$", 0.15)
    test_number(p3, 3, (2, 5), "test1", 0.1)
    test_log_slider(p4, 1, (1, 10), 10, "test2")

    # one formatter should be set
    assert isinstance(t.formatters, dict)
    assert len(t.formatters) == 4
    assert all(t.formatters[k] is None for k in [a, b, d])
    assert isinstance(
        t.formatters[c], bokeh.models.formatters.PrintfTickFormatter
    )

    # test mix tuple and parameters
    t = DynamicParam(
        params={
            a: (1, 0, 5),
            b: (1, 1, 10, 10, None, "test3", "log"),
            c: param.Boolean(default=True, label="test4"),
            d: param.ObjectSelector(
                default=5, objects=[1, 2, 3, 4, 5], label="test5"
            ),
            e: param.Number(
                default=6.1, softbounds=(1.1, 10.1), label="test6"
            ),
            f: param.Integer(default=6, softbounds=(1, None), label="test7"),
        },
        use_latex=False,
    )
    p1 = getattr(t.param, "dyn_param_0")
    p2 = getattr(t.param, "dyn_param_1")
    p3 = getattr(t.param, "dyn_param_2")
    p4 = getattr(t.param, "dyn_param_3")
    p5 = getattr(t.param, "dyn_param_4")
    p6 = getattr(t.param, "dyn_param_5")
    test_number(p1, 1, (0, 5), "a", 0.125)
    test_log_slider(p2, 1, (1, 10), 10, "test3")
    assert isinstance(p3, param.Boolean)
    assert p3.default is True
    assert p3.label == "test4"
    assert isinstance(p4, param.ObjectSelector)
    assert p4.label == "test5"
    assert p4.default == 5
    assert isinstance(p5, param.Number)
    assert p5.default == 6.1
    assert p5.softbounds == (1.1, 10.1)
    assert p5.label == "test6"
    assert isinstance(p6, param.Integer)
    assert p6.default == 6
    assert p6.label == "test7"

    r = {a: 1, b: 1, c: True, d: 5, e: 6.1, f: 6}
    assert t.read_parameters() == r


def test_DynamicParam_symbolic_parameters():
    # verify that we can pass symbolic numbers, which will then be converted
    # to float numbers

    a, b, c = symbols("a, b, c")

    # test _tuple_to_dict
    t = DynamicParam(
        params={
            a: (Integer(1), 0, 5),
            b: (2, Float(1.5), 4.5, Integer(20)),
            c: (3 * pi / 2, Rational(2, 3), Float(5), 30, None, "test1"),
        },
        use_latex=False,
    )
    p1 = getattr(t.param, "dyn_param_0")
    p2 = getattr(t.param, "dyn_param_1")
    p3 = getattr(t.param, "dyn_param_2")

    def test_number(p, d, sb):
        assert isinstance(p, param.Number)
        assert np.isclose(p.default, d) and isinstance(p.default, float)
        assert p.softbounds == sb
        assert all(isinstance(t, float) for t in p.softbounds)
        assert isinstance(p.step, float)

    test_number(p1, 1, (0, 5))
    test_number(p2, 2, (1.5, 4.5))
    test_number(p3, 1.5 * np.pi, (2 / 3, 5))


def test_iplot(panel_options):
    # verify that the correct widgets are created, with the appropriate labels

    bm = bokeh.models
    a, b, c, d = symbols("a, b, c, d")
    x, y, u, v = symbols("x, y, u, v")

    t = plot(
        (a + b + c + d) * cos(x),
        (x, -5, 5),
        params={
            a: (2, 1, 3, 5),
            b: (3, 2, 4000, 10, None, "label", "log"),
            c: param.Number(0.15, softbounds=(0, 1), label="test", step=0.025),
            # TODO: if I remove the following label, the tests are going to
            # fail: it would use the label "test5"... How is it possible?
            d: param.Integer(1, softbounds=(0, 10), label="d"),
            y: param.Integer(1, softbounds=(0, None)),
            u: param.Boolean(default=True),
            v: param.ObjectSelector(default=2, objects=[1, 2, 3, 4]),
        },
        layout="tb",
        ncols=2,
        use_latex=False,
        **panel_options
    )

    # no latex in labels
    p1 = getattr(t.param, "dyn_param_0")
    p2 = getattr(t.param, "dyn_param_1")
    p3 = getattr(t.param, "dyn_param_2")
    p4 = getattr(t.param, "dyn_param_3")

    assert p1.label == "a"
    assert p2.label == "label"
    assert p3.label == "test"
    assert p4.label == "d"

    # there are 7 parameters in this plot
    assert len(t.mapping) == 7

    # c1 wraps the controls, c2 wraps the plot
    c1, c2 = t.show().get_root().children
    gridbox = c1.children[0].children[0]
    assert isinstance(gridbox.children[0][0], bm.Slider)
    assert isinstance(gridbox.children[1][0].children[1], bm.Slider)
    assert isinstance(gridbox.children[2][0], bm.Slider)
    assert isinstance(gridbox.children[3][0], bm.Slider)
    assert isinstance(gridbox.children[4][0], bm.Spinner)
    assert isinstance(gridbox.children[5][0], bm.Checkbox)
    assert isinstance(gridbox.children[6][0], bm.Select)

    # test that the previous class-attribute associated to the previous
    # parameters are cleared in a new instance
    current_params = [
        k for k in InteractivePlot.__dict__.keys() if "dyn_param_" in k
    ]
    assert len(current_params) == 7

    t = plot(
        (a + b + c) * cos(x),
        (x, -5, 5),
        params={
            a: (2, 1, 3, 5),
            b: (3, 2, 4000, 10),
            c: param.Number(0.15, softbounds=(0, 1), label="test", step=0.025),
        },
        layout="tb",
        ncols=2,
        use_latex=True,
        **panel_options
    )

    # there are 3 parameters in this plot
    assert len(t.mapping) == 3

    # latex in labels
    p1 = getattr(t.param, "dyn_param_0")
    p2 = getattr(t.param, "dyn_param_1")
    p3 = getattr(t.param, "dyn_param_2")

    assert p1.label == "$$a$$"
    assert p2.label == "$$b$$"
    assert p3.label == "test"

    t = plot(
        (a + b) * cos(x),
        (x, -5, 5),
        params={
            a: (1, 0, 5),
            b: (1, 1, 10, 10, None, "test3", "log"),
        },
        use_latex=False,
        **panel_options
    )

    new_params = [
        k for k in InteractivePlot.__dict__.keys() if "dyn_param_" in k
    ]
    assert len(new_params) == 2


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
            x: (2, 0, 4),
            y: (200, 1, 1000, 10, formatter, "y", "log"),
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

    assert all(w[k].format is None for k in [x, z])
    assert isinstance(w[y].format, bokeh.models.formatters.PrintfTickFormatter)


@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
def test_iplot_sum_1(panel_options):
    # verify that it is possible to add together different instances of
    # InteractivePlot (as well as Plot instances), provided that the same
    # parameters are used.

    x, u = symbols("x, u")

    params = {u: (1, 0, 2)}
    p1 = plot(
        cos(u * x), (x, -5, 5),
        params=params,
        backend=MB,
        xlabel="x1",
        ylabel="y1",
        title="title 1",
        legend=True,
        pane_kw={"width": 500},
        **panel_options
    )
    p2 = plot(
        sin(u * x), (x, -5, 5),
        params=params,
        backend=MB,
        xlabel="x2",
        ylabel="y2",
        title="title 2",
        **panel_options
    )
    p3 = plot(
        sin(x) * cos(x), (x, -5, 5),
        backend=MB,
        adaptive=False,
        n=50,
        is_point=True,
        is_filled=True,
        line_kw=dict(marker="^"),
        **panel_options
    )
    p = p1 + p2 + p3

    assert isinstance(p, InteractivePlot)
    assert isinstance(p.backend, MB)
    assert p.backend.title == "title 1"
    assert p.backend.xlabel == "x1"
    assert p.backend.ylabel == "y1"
    assert p.backend.legend
    assert len(p.backend.series) == 3
    assert len([s for s in p.backend.series if s.is_interactive]) == 2
    assert len([s for s in p.backend.series if not s.is_interactive]) == 1
    assert p.pane_kw == {"width": 500}


def test_iplot_sum_2(panel_options):
    # verify that it is not possible to add together different instances of
    # InteractivePlot when they are using different parameters

    x, u, v = symbols("x, u, v")

    p1 = plot(
        cos(u * x), (x, -5, 5),
        params={u: (1, 0, 1)},
        backend=MB,
        xlabel="x1",
        ylabel="y1",
        title="title 1",
        legend=True,
        **panel_options
    )
    p2 = plot(
        sin(u * x) + v, (x, -5, 5),
        params={u: (1, 0, 1), v: (0, -2, 2)},
        backend=MB,
        xlabel="x2",
        ylabel="y2",
        title="title 2",
        **panel_options
    )
    raises(ValueError, lambda: p1 + p2)


@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
def test_iplot_sum_3(panel_options):
    # verify that the resulting iplot's backend is of the same type as the
    # original

    x, u = symbols("x, u")

    def func(B):
        params = {u: (1, 0, 2)}
        p1 = plot(
            cos(u * x), (x, -5, 5),
            params=params,
            backend=B,
            xlabel="x1",
            ylabel="y1",
            title="title 1",
            legend=True,
            **panel_options
        )
        p2 = plot(
            sin(u * x), (x, -5, 5),
            params=params,
            backend=B,
            xlabel="x2",
            ylabel="y2",
            title="title 2",
            **panel_options
        )
        p = p1 + p2
        assert isinstance(p.backend, B)

    func(MB)
    func(BB)
    func(PB)


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
