from appdirs import user_data_dir
import os
import json
from mergedeep import merge
import warnings
from inspect import currentframe

appname = "spb"
cfg_file = "config.json"
cfg_dir = user_data_dir(appname)
os.makedirs(cfg_dir, exist_ok=True)
file_path = os.path.join(cfg_dir, cfg_file)


def _hardcoded_defaults():
    # Hardcoded default values
    return dict(
        # Find more Plotly themes at the following page:
        # https://plotly.com/python/templates/
        plotly={"theme": "plotly_dark"},
        # Find more Bokeh themes at the following page:
        # https://docs.bokeh.org/en/latest/docs/reference/themes.html
        bokeh={"theme": "dark_minimal", "sizing_mode": "stretch_width"},
        k3d={
            "bg_color": 3620427,
            "grid_color": 0x888888,
            "label_color": 0xDDDDDD,
        },
        matplotlib={
            "axis_center": None,  # "auto"
            "grid": True,
            "use_jupyterthemes": False,
            "jupytertheme": None,
        },
        backend_2D="matplotlib",
        backend_3D="matplotlib",
        complex={
            "modules": None,     # "mpmath"
            "coloring": "a"
        }
    )


def reset():
    """Restore original settings."""
    set_defaults(_hardcoded_defaults())


def _load_settings():
    """Load settings and inject the names into the current namespace."""
    frame = currentframe()

    cfg = dict()
    if os.path.exists(file_path):
        with open(file_path) as f:
            cfg = json.load(f)

    default_cfg = _hardcoded_defaults()

    # Because the user can directly change the configuration file, we need to assure
    # that all the necessary options are present (maybe, the user deleted something
    # accidentally)
    cfg = merge({}, default_cfg, cfg)
    frame.f_globals["cfg"] = cfg

    # check that the chosen backends are available
    backends_2D = ["plotly", "bokeh", "matplotlib"]
    backends_3D = ["plotly", "matplotlib", "k3d", "mayavi"]

    def check_backend(k, backends):
        if cfg[k] not in backends:
            # restore hardcoded values in order to be able to load the module
            # the next time
            reset()

            raise ValueError(
                "`{}` must be one of the following ".format(k)
                + "values: {}\n".format(backends)
                + "Received: = '{}'\n".format(cfg[k])
                + "Reset config file to hardcoded default values: done."
            )

    check_backend("backend_2D", backends_2D)
    check_backend("backend_3D", backends_3D)

    # load the selected backends
    if cfg["backend_2D"] == "plotly":
        from spb.backends.plotly import PlotlyBackend as TWO_D_B
    elif cfg["backend_2D"] == "bokeh":
        from spb.backends.bokeh import BokehBackend as TWO_D_B
    elif cfg["backend_2D"] == "matplotlib":
        from spb.backends.matplotlib import MatplotlibBackend as TWO_D_B
    elif cfg["backend_2D"] == "k3d":
        from spb.backends.k3d import K3DBackend as TWO_D_B
    elif cfg["backend_2D"] == "mayavi":
        from spb.backends.mayavi import MayaviBackend as TWO_D_B

    if cfg["backend_2D"] == cfg["backend_3D"]:
        THREE_D_B = TWO_D_B
    else:
        if cfg["backend_3D"] == "plotly":
            from spb.backends.plotly import PlotlyBackend as THREE_D_B
        elif cfg["backend_3D"] == "matplotlib":
            from spb.backends.matplotlib import MatplotlibBackend as THREE_D_B
        elif cfg["backend_3D"] == "k3d":
            from spb.backends.k3d import K3DBackend as THREE_D_B
        elif cfg["backend_3D"] == "mayavi":
            from spb.backends.mayavi import MayaviBackend as THREE_D_B

    frame.f_globals["TWO_D_B"] = TWO_D_B
    frame.f_globals["THREE_D_B"] = THREE_D_B


def set_defaults(cfg):
    """Set the default options for the plotting backends.

    Parameters
    ==========
        cfg : dict
            Dictionary containing the new values

    Example
    =======

    Change the default 2D plotting backend to BokehBackend and set the its
    theme to "night_sky"

    .. code-block:: python
        from spb.defaults import cfg, set_defaults
        ## to visualize the available options
        # print(cfg)
        cfg["backend_2D"] = "bokeh"
        cfg["bokeh"]["theme"] = "night_sky"
        set_defaults(cfg)

    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
        warnings.warn("Successfully written settings to {}".format(file_path))

    _load_settings()


_load_settings()
