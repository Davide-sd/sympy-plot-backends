from sympy import symbols, cos, sin, Tuple, Plane
from sympy.external import import_module
from spb.interactive import (
    iplot, DynamicParam, MyList,
    InteractivePlot, create_widgets
)
from spb.functions import plot
from spb.series import InteractiveSeries
from spb.backends.plotly import PB
from spb.backends.matplotlib import MB
from pytest import raises

np = import_module('numpy', catch=(RuntimeError,))
param = import_module(
    'param',
    min_module_version='1.11.0',
    catch=(RuntimeError,))
pn = import_module(
    'panel',
    min_module_version='0.12.0',
    catch=(RuntimeError,))
bokeh = import_module(
    'bokeh',
    import_kwargs={'fromlist':['models']},
    min_module_version='2.3.0',
    catch=(RuntimeError,))

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

    test_number(p1, 1, (0, 5), "$a$", 0.125)
    test_number(p2, 2, (1.5, 4.5), "$b$", 0.15)
    test_number(p3, 3, (2, 5), "test1", 0.1)
    test_log_slider(p4, 1, (1, 10), 10, "test2")

    # one formatter should be set
    assert isinstance(t.formatters, dict)
    assert len(t.formatters) == 4
    assert all(t.formatters[k] is None for k in [a, b, d])
    assert isinstance(t.formatters[c], bokeh.models.formatters.PrintfTickFormatter)

    # test mix tuple and parameters
    t = DynamicParam(
        params={
            a: (1, 0, 5),
            b: (1, 1, 10, 10, None, "test3", "log"),
            c: param.Boolean(default=True, label="test4"),
            d: param.ObjectSelector(default=5, objects=[1, 2, 3, 4, 5], label="test5"),
            e: param.Number(default=6.1, softbounds=(1.1, 10.1), label="test6"),
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
    assert p3.default == True
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

    # raise error because of an invalid formatter. The formatter must be None
    # or an instance of bokeh.models.formatters.TickFormatter
    raises(TypeError, lambda: DynamicParam(
        params={
            a: (1, 0, 5),
            b: (2, 1.5, 4.5, 20),
            c: (3, 2, 5, 30, True, "test1"),
            d: (1, 1, 10, 10, None, "test2", "log"),
        },
        use_latex=True,
    ))

def test_iplot():
    bm = bokeh.models
    a, b, c, d = symbols("a, b, c, d")
    x, y, u, v = symbols("x, y, u, v")

    t = iplot(
        ((a + b + c + d) * cos(x), (x, -5, 5)),
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
        show=False,
        layout="tb",
        ncols=2,
        use_latex=False
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
    assert isinstance(gridbox.children[5][0], bm.CheckboxGroup)
    assert isinstance(gridbox.children[6][0], bm.Select)

    # test that the previous class-attribute associated to the previous
    # parameters are cleared in a new instance
    current_params = [k for k in InteractivePlot.__dict__.keys() if "dyn_param_" in k]
    assert len(current_params) == 7

    t = iplot(
        ((a + b + c) * cos(x), (x, -5, 5)),
        params={
            a: (2, 1, 3, 5),
            b: (3, 2, 4000, 10),
            c: param.Number(0.15, softbounds=(0, 1), label="test", step=0.025),
        },
        show=False,
        layout="tb",
        ncols=2,
        use_latex=True
    )

    # there are 3 parameters in this plot
    assert len(t.mapping) == 3

    # latex in labels
    p1 = getattr(t.param, "dyn_param_0")
    p2 = getattr(t.param, "dyn_param_1")
    p3 = getattr(t.param, "dyn_param_2")

    assert p1.label == "$a$"
    assert p2.label == "$b$"
    assert p3.label == "test"

    t = iplot(
        ((a + b) * cos(x), (x, -5, 5)),
        params={
            a: (1, 0, 5),
            b: (1, 1, 10, 10, None, "test3", "log"),
        },
        use_latex=False,
        show=False,
    )

    new_params = [k for k in InteractivePlot.__dict__.keys() if "dyn_param_" in k]
    assert len(new_params) == 2


def test_create_widgets():
    x, y, z = symbols("x:z")

    w = create_widgets({
        x: (2, 0, 4),
        y: (200, 1, 1000, 10, None, "$y$", "log"),
        z: param.Integer(3, softbounds=(3, 10), label="n")
    }, use_latex = True)

    assert isinstance(w, dict)
    assert len(w) == 3
    assert isinstance(w[x], pn.widgets.FloatSlider)
    assert isinstance(w[y], pn.widgets.DiscreteSlider)
    assert isinstance(w[z], pn.widgets.IntSlider)
    assert w[x].name == "$x$"
    assert w[y].name == "$y$"
    assert w[z].name == "n"

    formatter = bokeh.models.formatters.PrintfTickFormatter(format="%.4f")
    w = create_widgets({
        x: (2, 0, 4),
        y: (200, 1, 1000, 10, formatter, "y", "log"),
        z: param.Integer(3, softbounds=(3, 10), label="n")
    }, use_latex = False)

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


def test_interactiveseries():
    # test for the instantiation of InteractiveSeries
    from sympy.vector import CoordSys3D

    N = CoordSys3D("N")
    i, j, k = N.base_vectors()
    x, y, z = N.base_scalars()
    a, b, c = symbols("a:c")
    v1 = -a * sin(y) * i + b * cos(x) * j
    m1 = v1.to_matrix(N)
    m1 = m1[:-1]
    l1 = list(m1)
    v2 = -a * sin(y) * i + b * cos(x) * j + c * cos(z) * k
    m2 = v2.to_matrix(N)
    l2 = list(m2)

    def test_vector(v, ranges, params, expr, label, symbol, shape, n=10):
        t = iplot((v, *ranges), params=params, n=n, backend=PB, show=False)

        s = t.backend.series[0]
        assert isinstance(s, InteractiveSeries)
        assert s.expr == expr
        assert s.label == label
        assert len(s.ranges) == len(ranges)
        assert s.ranges[symbol].shape == shape
        if len(ranges) == 2:
            assert s.is_2Dvector
            assert not s.is_3Dvector
        else:
            assert not s.is_2Dvector
            assert s.is_3Dvector
        
        # verify that the backend is able to run the `_update_interactive`
        # method.
        new_params = {k: v[0] for k, v in params.items()}
        t._backend._update_interactive(new_params)

    # 2D vectors
    params = {
        a: (2, 0, 3),
        b: (3, 1, 4),
    }
    ranges = (x, -5, 5), (y, -4, 4)
    test_vector(
        v1, ranges, params, Tuple(-a * sin(N.y), b * cos(N.x)), str(v1),
        N.x, (10, 10)
    )
    test_vector(
        m1, ranges, params, Tuple(-a * sin(y), b * cos(x)), str(tuple(m1)),
        x, (10, 10)
    )
    test_vector(
        l1, ranges, params, Tuple(-a * sin(y), b * cos(x)), str(tuple(l1)),
        x, (8, 8), 8
    )

    # 3D vectors
    params = {
        a: (2, 0, 3),
        b: (3, 1, 4),
        c: (4, 2, 5),
    }
    ranges = (x, -5, 5), (y, -4, 4), (z, -6, 6)
    test_vector(
        v2,
        ranges,
        params,
        Tuple(-a * sin(N.y), b * cos(N.x), c * cos(N.z)),
        str(v2),
        N.x,
        (10, 10, 10),
    )
    test_vector(
        m2,
        ranges,
        params,
        Tuple(-a * sin(y), b * cos(x), c * cos(z)),
        str(m2),
        x,
        (10, 10, 10),
    )
    test_vector(
        l2,
        ranges,
        params,
        Tuple(-a * sin(y), b * cos(x), c * cos(z)),
        str(tuple(l2)),
        x,
        (10, 10, 10),
    )

    # Sliced 3D vectors: single slice
    v3 = a * z * i + b * y * j + c * x * k
    t = iplot(
        (v3, *ranges),
        params=params,
        n1=5,
        n2=6,
        n3=7,
        slice=Plane((1, 2, 3), (1, 0, 0)),
        backend=PB,
        show=False,
    )
    assert len(t.backend.series) == 1
    s = t.backend.series[0]
    assert isinstance(s, InteractiveSeries)
    assert s.is_3Dvector
    assert s.is_slice
    xx, yy, zz, uu, vv, ww = s.get_data()
    assert all([t.shape == (6, 7) for t in [xx, yy, zz, uu, vv, ww]])
    assert np.all(xx == 1)
    assert (np.min(yy.flatten()) == -4) and (np.max(yy.flatten()) == 4)
    assert (np.min(zz.flatten()) == -6) and (np.max(zz.flatten()) == 6)

    # Sliced 3D vectors: multiple slices. Test that each slice creates a
    # corresponding series
    t = iplot(
        (v3, *ranges),
        params=params,
        n1=5,
        n2=6,
        n3=7,
        slice=[
            Plane((1, 2, 3), (1, 0, 0)),
            Plane((1, 2, 3), (0, 1, 0)),
            Plane((1, 2, 3), (0, 0, 1)),
        ],
        backend=PB,
        show=False,
    )
    assert len(t.backend.series) == 3
    assert all([isinstance(s, InteractiveSeries) for s in t.backend.series])
    assert all([s.is_3Dvector for s in t.backend.series])
    assert all([s.is_slice for s in t.backend.series])
    xx1, yy1, zz1, uu1, vv1, ww1 = t.backend.series[0].get_data()
    xx2, yy2, zz2, uu2, vv2, ww2 = t.backend.series[1].get_data()
    xx3, yy3, zz3, uu3, vv3, ww3 = t.backend.series[2].get_data()
    assert all([t.shape == (6, 7) for t in [xx1, yy1, zz1, uu1, vv1, ww1]])
    assert all([t.shape == (7, 5) for t in [xx2, yy2, zz2, uu2, vv2, ww2]])
    assert all([t.shape == (6, 5) for t in [xx3, yy3, zz3, uu3, vv3, ww3]])
    assert np.all(xx1 == 1)
    assert (np.min(yy1.flatten()) == -4) and (np.max(yy1.flatten()) == 4)
    assert (np.min(zz1.flatten()) == -6) and (np.max(zz1.flatten()) == 6)
    assert np.all(yy2 == 2)
    assert (np.min(xx2.flatten()) == -5) and (np.max(xx2.flatten()) == 5)
    assert (np.min(zz2.flatten()) == -6) and (np.max(zz2.flatten()) == 6)
    assert np.all(zz3 == 3)
    assert (np.min(xx3.flatten()) == -5) and (np.max(xx3.flatten()) == 5)
    assert (np.min(yy3.flatten()) == -4) and (np.max(yy3.flatten()) == 4)

def test_iplot_sum_1():
    # verify that it is possible to add together different instances of
    # InteractivePlot (as well as Plot instances), provided that the same
    # parameters are used.

    x, u = symbols("x, u")

    params = {
        u: (1, 0, 2)
    }
    p1 = iplot(
        (cos(u * x), (x, -5, 5)),
        params = params,
        backend = MB,
        xlabel = "x1",
        ylabel = "y1",
        title = "title 1",
        legend=True,
        show = False
    )
    p2 = iplot(
        (sin(u * x), (x, -5, 5)),
        params = params,
        backend = MB,
        xlabel = "x2",
        ylabel = "y2",
        title = "title 2",
        show = False
    )
    p3 = plot(sin(x)*cos(x), (x, -5, 5), backend=MB,
        adaptive=False, n=50,
        is_point=True, is_filled=True,
        line_kw=dict(marker="^"), show=False)
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

def test_iplot_sum_2():
    # verify that it is not possible to add together different instances of
    # InteractivePlot when they are using different parameters

    x, u, v = symbols("x, u, v")

    p1 = iplot(
        (cos(u * x), (x, -5, 5)),
        params = {
            u: (1, 0, 1)
        },
        backend = MB,
        xlabel = "x1",
        ylabel = "y1",
        title = "title 1",
        legend=True,
        show = False
    )
    p2 = iplot(
        (sin(u * x) + v, (x, -5, 5)),
        params = {
            u: (1, 0, 1),
            v: (0, -2, 2)
        },
        backend = MB,
        xlabel = "x2",
        ylabel = "y2",
        title = "title 2",
        show = False
    )
    p = p1 + p2
    raises(KeyError, lambda: p.show())
