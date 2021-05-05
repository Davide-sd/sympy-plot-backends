from sympy import symbols, cos, sin
import param
import panel as pn
import bokeh.models as bm
from spb.interactive import iplot, DynamicParam, MyList

def test_DynamicParam():
    a, b, c, d, e, f = symbols("a, b, c, d, e, f")

    # test _tuple_to_dict
    t = DynamicParam(params = {
        a: (1, (0, 5)),
        b: (2, (1.5, 4.5), 20),
        c: (3, (2, 5), 30, "test1"),
        d: (1, (1, 10), 10, "test2", "log"),
    }, use_latex=False)
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

    # test mix tuple and parameters
    t = DynamicParam(params = {
        a: (1, (0, 5)),
        b: (1, (1, 10), 10, "test3", "log"),
        c: param.Boolean(default=True, label="test4"),
        d: param.ObjectSelector(default=5, objects=[1, 2, 3, 4, 5], label="test5"),
        e: param.Number(default=6.1, softbounds=(1.1, 10.1), label="test6"),
        f: param.Integer(default=6, softbounds=(1, None), label="test7"),
    }, use_latex=False)
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

    r = { a: 1, b: 1, c: True, d: 5, e: 6.1, f: 6 }
    assert t.read_parameters() == r


def test_iplot():
    a, b, c, d = symbols("a, b, c, d")
    x, y, u, v = symbols("x, y, u, v")

    t = iplot(
        ((a + b + c + d) * cos(x), (x, -5, 5)),
        params = {
            a: (2, (1, 3), 5),
            b: (3, (2, 4000), 10, "label", "log"),
            c: param.Number(0.15, softbounds=(0, 1), label="test", step=0.025),
            d: param.Integer(1, softbounds=(0, 10)),
            y: param.Integer(1, softbounds=(0, None)),
            u: param.Boolean(default=True),
            v: param.ObjectSelector(default=2, objects=[1, 2, 3, 4]),
        }, show=False, layout="tb", ncols=2
    )

    # there are 4 parameters in this plot
    assert len(t.mapping) == 7
    
    # c1 wraps the controls, c2 wraps the plot
    c1, c2 = t.show().get_root().children
    gridbox = c1.children[0].children[0]
    print(type(gridbox.children[0][0]))
    assert isinstance(gridbox.children[0][0], bm.Slider)
    assert isinstance(gridbox.children[1][0].children[1], bm.Slider)
    assert isinstance(gridbox.children[2][0], bm.Slider)
    assert isinstance(gridbox.children[3][0], bm.Slider)
    assert isinstance(gridbox.children[4][0], bm.Slider)
    assert isinstance(gridbox.children[5][0], bm.CheckboxGroup)
    assert isinstance(gridbox.children[6][0], bm.Select)

    
