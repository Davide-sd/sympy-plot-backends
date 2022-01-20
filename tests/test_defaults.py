from spb.defaults import cfg, set_defaults, reset
from spb.backends.plotly import PB
from spb.backends.bokeh import BB
from spb.backends.matplotlib import MB
from spb.functions import plot, plot3d
from sympy import symbols, sin
from pytest import raises


def test_cfg():
    assert isinstance(cfg, dict)
    assert "backend_2D" in cfg.keys()
    assert "backend_3D" in cfg.keys()

    assert "matplotlib" in cfg.keys()
    assert "axis_center" in cfg["matplotlib"].keys()
    assert "grid" in cfg["matplotlib"].keys()

    assert "plotly" in cfg.keys()
    assert "theme" in cfg["plotly"].keys()
    assert isinstance(cfg["plotly"]["theme"], str)

    assert "bokeh" in cfg.keys()
    assert "theme" in cfg["bokeh"].keys()
    assert isinstance(cfg["bokeh"]["theme"], str)

    assert "k3d" in cfg.keys()
    assert "bg_color" in cfg["k3d"].keys()
    assert "grid_color" in cfg["k3d"].keys()
    assert "label_color" in cfg["k3d"].keys()
    assert isinstance(cfg["k3d"]["bg_color"], int)
    assert isinstance(cfg["k3d"]["grid_color"], int)
    assert isinstance(cfg["k3d"]["label_color"], int)


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
