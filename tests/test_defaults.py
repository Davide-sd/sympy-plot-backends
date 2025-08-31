import json
import os
from pytest import raises, warns
from spb.defaults import cfg, set_defaults, reset
from sympy import symbols
from sympy.external import import_module


def test_cfg_keys():
    assert isinstance(cfg, dict)
    must_have_keys = [
        "backend_2D",
        "backend_3D",
        "use_repr",
        "matplotlib",
        "plotly",
        "k3d",
        "bokeh",
        "complex",
        "interactive",
        "plot3d",
        "plot_range",
        "bode",
    ]
    for k in must_have_keys:
        assert k in cfg.keys()


def test_use_repr():
    assert cfg["use_repr"] is False


def test_plot_range_keys():
    assert (
        ("min" in cfg["plot_range"].keys())
        and ("max" in cfg["plot_range"].keys())
    )


def test_plot3d_keys():
    assert (
        ("use_cm" in cfg["plot3d"].keys())
        and (cfg["plot3d"]["use_cm"] is False)
    )


def test_cfg_matplotlib_keys():
    matplotlib_keys = [
        "axis_center", "grid", "show_minor_grid", "use_latex", "update_event"]
    for k in matplotlib_keys:
        assert k in cfg["matplotlib"].keys()
    assert isinstance(cfg["matplotlib"]["grid"], bool)
    assert isinstance(cfg["matplotlib"]["show_minor_grid"], bool)
    assert isinstance(cfg["matplotlib"]["update_event"], bool)


def test_cfg_plotly_keys():
    must_have_keys = [
        "theme", "grid", "use_latex", "update_event", "show_minor_grid"]
    for k in must_have_keys:
        assert k in cfg["plotly"].keys()
    assert isinstance(cfg["plotly"]["theme"], str)
    assert isinstance(cfg["plotly"]["grid"], bool)
    assert isinstance(cfg["plotly"]["show_minor_grid"], bool)
    assert isinstance(cfg["plotly"]["update_event"], bool)


def test_cfg_bokeh_keys():
    bokeh_keys = [
        "theme",
        "sizing_mode",
        "show_minor_grid",
        "width",
        "height",
        "minor_grid_line_alpha",
        "minor_grid_line_dash",
        "grid",
        "use_latex",
        "update_event"
    ]
    for k in bokeh_keys:
        assert k in cfg["bokeh"].keys()
    assert isinstance(cfg["bokeh"]["sizing_mode"], str)
    assert isinstance(cfg["bokeh"]["width"], int)
    assert isinstance(cfg["bokeh"]["height"], int)
    assert isinstance(cfg["bokeh"]["sizing_mode"], str)
    assert isinstance(cfg["bokeh"]["grid"], bool)
    assert isinstance(cfg["bokeh"]["show_minor_grid"], bool)
    assert isinstance(cfg["bokeh"]["minor_grid_line_alpha"], (float, int))
    assert isinstance(cfg["bokeh"]["minor_grid_line_dash"], (list, tuple))
    assert isinstance(cfg["bokeh"]["update_event"], bool)


def test_cfg_k3d_keys():
    k3d_keys = [
        "bg_color",
        "grid_color",
        "label_color",
        "use_latex",
    ]
    for k in k3d_keys:
        assert k in cfg["k3d"].keys()
        assert isinstance(cfg["k3d"][k], int)

    assert "grid" in cfg["k3d"].keys()
    assert isinstance(cfg["k3d"]["grid"], bool)
    assert "camera_mode" in cfg["k3d"].keys()


def test_cfg_interactive_keys():
    must_have_keys = ["phase_unit", "freq_unit"]
    for k in must_have_keys:
        assert k in cfg["bode"].keys()
    assert isinstance(cfg["bode"]["phase_unit"], str)
    assert isinstance(cfg["bode"]["freq_unit"], str)
    assert cfg["bode"]["phase_unit"] == "rad"
    assert cfg["bode"]["freq_unit"] == "rad/sec"


def test_cfg_bode_keys():
    must_have_keys = ["use_latex", "throttled", "servable", "theme", "module"]
    for k in must_have_keys:
        assert k in cfg["interactive"].keys()
    assert isinstance(cfg["interactive"]["use_latex"], bool)
    assert isinstance(cfg["interactive"]["throttled"], bool)
    assert isinstance(cfg["interactive"]["servable"], bool)
    assert isinstance(cfg["interactive"]["theme"], str)
    assert isinstance(cfg["interactive"]["module"], str)


def test_set_defaults():
    appdirs = import_module("appdirs", min_module_version="1.4.4")

    # NOTE: sadly, it's impossible to run plot commands to test if
    # the changes has been applied, as that requires the kernel to
    # restart. Here, we just check that the configuration file has
    # been updated.

    def get_current_setting():
        appname = "spb"
        cfg_file = "config.json"
        cfg_dir = appdirs.user_data_dir(appname)
        file_path = os.path.join(cfg_dir, cfg_file)

        loaded_cfg = dict()
        if os.path.exists(file_path):
            with open(file_path) as f:
                loaded_cfg = json.load(f)
        return loaded_cfg

    x, y = symbols("x, y")

    # changing backends should be a smooth operation
    cfg["backend_2D"] = "bokeh"
    cfg["backend_3D"] = "matplotlib"
    with warns(UserWarning, match="Successfully written settings"):
        set_defaults(cfg)
    loaded_cfg = get_current_setting()
    assert loaded_cfg["backend_2D"] == "bokeh"
    assert loaded_cfg["backend_3D"] == "matplotlib"

    # wrong backends settings -> reset to default settings
    cfg["backend_2D"] = "k3d"
    with warns(UserWarning, match="Successfully written settings"):
        raises(ValueError, lambda: set_defaults(cfg))
    loaded_cfg = get_current_setting()
    assert loaded_cfg["backend_2D"] == "matplotlib"


def test_reset():
    reset()
