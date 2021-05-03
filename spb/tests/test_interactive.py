from sympy import symbols, cos, sin
import param
from spb.interactive import iplot

def test_iplot():
    a, b, c, d = symbols("a, b, c, d")
    x, y, u, v = symbols("x, y, u, v")

    def test_slider(control, val, bounds, label, step):
        assert control.default == val
        assert control.softbounds == bounds
        assert control.label == label
        assert control.step == step

    t = iplot(
        ((a + b + c + d) * cos(x), (x, -5, 5)),
        params = {
            a: (1, (0, 2)),
            b: (2, (1, 3), 5),
            c: (3, (2, 4), 10, "label"),
            d: param.Number(0.15, softbounds=(0, 1), label="test", step=0.025)
        },
        show = False
    )

    # there are 4 parameters in this plot
    assert len(t.mapping) == 4
    
    # test DynamicPar._tuple_to_dict
    c1 = getattr(t.param, "dyn_param_0")
    c2 = getattr(t.param, "dyn_param_1")
    c3 = getattr(t.param, "dyn_param_2")
    c4 = getattr(t.param, "dyn_param_3")
    test_slider(c1, 1, (0, 2), "$a$", 0.05)
    test_slider(c2, 2, (1, 3), "$b$", 0.4)
    test_slider(c3, 3, (2, 4), "label", 0.2)
    # test param.Number
    test_slider(c4, 0.15, (0, 1), "test", 0.025)

    r = { a: 1, b: 2, c: 3, d: 0.15 }
    assert t.read_parameters() == r


