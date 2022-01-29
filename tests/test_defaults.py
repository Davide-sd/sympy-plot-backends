from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import sin
from spb.defaults import cfg, set_defaults, reset
from spb.backends.bokeh import BB
from spb.backends.matplotlib import MB
from spb.functions import plot, plot3d
from pytest import raises


def test_cfg_keys():
    assert isinstance(cfg, dict)
    must_have_keys = ["backend_2D", "backend_3D", "matplotlib", "plotly",
        "k3d", "bokeh", "complex", "interactive"]
    for k in must_have_keys:
        assert k in cfg.keys()


def test_cfg_matplotlib_keys():
    matplotlib_keys = ["axis_center", "grid", "show_minor_grid"]
    for k in matplotlib_keys:
        assert k in cfg["matplotlib"].keys()
    assert isinstance(cfg["matplotlib"]["grid"], bool)
    assert isinstance(cfg["matplotlib"]["show_minor_grid"], bool)


def test_cfg_plotly_keys():
    assert "theme" in cfg["plotly"].keys()
    assert isinstance(cfg["plotly"]["theme"], str)


def test_cfg_bokeh_keys():
    bokeh_keys = ["theme", "sizing_mode", "update_event", "show_minor_grid",
        "minor_grid_line_alpha", "minor_grid_line_dash"]
    for k in bokeh_keys:
        assert k in cfg["bokeh"].keys()
    assert isinstance(cfg["bokeh"]["sizing_mode"], str)
    assert isinstance(cfg["bokeh"]["update_event"], bool)
    assert isinstance(cfg["bokeh"]["show_minor_grid"], bool)
    assert isinstance(cfg["bokeh"]["minor_grid_line_alpha"], (float, int))
    assert isinstance(cfg["bokeh"]["minor_grid_line_dash"], (list, tuple))


def test_cfg_k3d_keys():
    k3d_keys = ["bg_color", "grid_color", "label_color"]
    for k in k3d_keys:
        assert k in cfg["k3d"].keys()
        assert isinstance(cfg["k3d"][k], int)


def test_cfg_interactive_keys():
    assert "use_latex" in cfg["interactive"]
    assert isinstance(cfg["interactive"]["use_latex"], bool)


def test_set_defaults():
    x, y = symbols("x, y")

    # changing backends should be a smooth operation
    cfg["backend_2D"] = "bokeh"
    cfg["backend_3D"] = "matplotlib"
    set_defaults(cfg)
    p = plot(sin(x), show=False)
    assert isinstance(p, BB)
    p = plot3d(sin(x ** 2 + y ** 2), show=False)
    assert isinstance(p, MB)

    # wrong backends settings -> reset to default settings
    cfg["backend_2D"] = "k3d"
    raises(ValueError, lambda: set_defaults(cfg))
    p = plot(sin(x), show=False)
    assert isinstance(p, MB)

    # reset original settings
    cfg["backend_2D"] = "bokeh"
    cfg["backend_3D"] = "matplotlib"
    set_defaults(cfg)
    reset()
    from spb.defaults import cfg as cfg2

    assert cfg2["backend_2D"] == "matplotlib"
    assert cfg2["backend_3D"] == "matplotlib"
