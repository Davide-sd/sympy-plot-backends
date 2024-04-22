from spb.animation import BaseAnimation
from spb.defaults import TWO_D_B, THREE_D_B, cfg
from spb.interactive import IPlot
from spb.interactive.bootstrap_spb import SymPyBootstrapTemplate
from spb.plotgrid import PlotGrid
from sympy.external import import_module
import warnings
from mergedeep import merge

param = import_module(
    'param',
    min_module_version='1.11.0',
    warn_not_installed=True)
pn = import_module(
    'panel',
    min_module_version='0.12.0',
    warn_not_installed=True)

pn.extension("mathjax", "plotly", sizing_mode="stretch_width")


class Animation(BaseAnimation, IPlot):
    def __init__(self, *series, **kwargs):
        self._servable = kwargs.pop("servable", cfg["interactive"]["servable"])
        self._pane_kw = kwargs.pop("pane_kw", dict())
        self._template = kwargs.pop("template", None)
        self._name = kwargs.pop("name", "")
        self._original_params = kwargs.get("params", {})
        self.merge = merge

        plotgrid = kwargs.get("plotgrid", None)
        if plotgrid:
            self._backend = plotgrid
            self._post_init_plotgrid(**kwargs)
        else:
            is_3D = all([s.is_3D for s in series])
            Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
            kwargs["is_iplot"] = True
            kwargs["imodule"] = "panel"
            self._backend = Backend(*series, **kwargs)
            self._post_init_plot(**kwargs)

        self._play_widget = pn.widgets.Player(
            value=0,
            start=0,
            end=self._animation_data.n_frames - 1,
            step=1,
            interval=int(1000 / self._animation_data.fps),
        )
        self._binding = pn.bind(self._update, self._play_widget)

    def _update(self, frame_idx):
        print("spb.animation.panel._update", frame_idx)
        self.update_animation(frame_idx)
        return self._backend.fig

    def _init_pane(self):
        """Here we wrap the figure exposed by the backend with a Pane, which
        allows to set useful properties.
        """
        # NOTE: If the following import statement was located at the
        # beginning of the file, there would be a circular import.
        from spb import KB, MB, BB, PB

        print("_init_pane")

        default_kw = {}
        if isinstance(self._backend, PB):
            print("Case 0")
            pane_func = pn.pane.Plotly
        elif (
            isinstance(self._backend, MB) or        # vanilla MB
            (
                hasattr(self._backend, "is_matplotlib_fig") and
                self._backend.is_matplotlib_fig     # plotgrid with all MBs
            )
        ):
            print("Case 1")
            # since we are using Jupyter and interactivity, it is useful to
            # activate ipympl interactive frame, as well as setting a lower
            # dpi resolution of the matplotlib image
            default_kw["dpi"] = 96
            # NOTE: the following must be set to False in order for the
            # example outputs to become visible on Sphinx.
            default_kw["interactive"] = False
            pane_func = pn.pane.Matplotlib
        elif isinstance(self._backend, BB):
            print("Case 2")
            pane_func = pn.pane.Bokeh
        elif isinstance(self._backend, KB):
            print("Case 3")
            # TODO: for some reason, panel is going to set width=0
            # if K3D-Jupyter is used.
            # Temporary workaround: create a Pane with a default width.
            # Long term solution: create a PR on panel to create a K3DPane
            # so that panel will automatically deal with K3D, in the same
            # way it does with Bokeh, Plotly, Matplotlib, ...
            default_kw["width"] = 800
            pane_func = pn.pane.panel
        else:
            print("Case 4")
            # here we are dealing with plotgrid of BB/PB/or mixed backend...
            # but not with plotgrids of MB
            # First, set the necessary data to create bindings for each
            # subplot
            self._backend.pre_set_bindings(
                [1], # anything but None
                [self._play_widget]
            )
            # Then, create the pn.GridSpec figure
            self.pane = self._backend.fig
            return
        kw = self.merge({}, default_kw, self._pane_kw)
        self.pane = pane_func(self._binding, **kw)
        print("self.pane", self.pane)

    def show(self):
        self._init_pane()

        if not self._servable:
            return pn.Column(self.pane, self._play_widget)

        return self._create_template(True)

    def _create_template(self, show=False):
        """Instantiate a template, populate it and serves it.

        Parameters
        ==========

        show : boolean
            If True, the template will be served on a new browser window.
            Otherwise, just return the template: ``show=False`` is used
            by the documentation to visualize servable applications.
        """
        if not show:
            self._init_pane()

        print("_create_template")

        # pn.theme was introduced with panel 1.0.0, before there was
        # pn.template.theme
        submodule = pn.theme if hasattr(pn, "theme") else pn.template.theme
        theme = submodule.DarkTheme
        if cfg["interactive"]["theme"] != "dark":
            theme = submodule.DefaultTheme
        default_template_kw = dict(title=self._name, theme=theme)

        if (self._template is None) or isinstance(self._template, dict):
            kw = self._template if isinstance(self._template, dict) else {}
            kw = self.merge(default_template_kw, kw)
            if len(self._name.strip()) == 0:
                kw.setdefault("show_header", False)
            template = SymPyBootstrapTemplate(**kw)
        elif isinstance(self._template, pn.template.base.BasicTemplate):
            template = self._template
        elif (isinstance(self._template, type) and
            issubclass(self._template, pn.template.base.BasicTemplate)):
            template = self._template(**default_template_kw)
        else:
            raise TypeError("`template` not recognized. It can either be a "
                "dictionary of keyword arguments to be passed to the default "
                "template, an instance of pn.template.base.BasicTemplate "
                "or a subclass of pn.template.base.BasicTemplate. Received: "
                "type(template) = %s" % type(self._template))

        template.main.append(pn.Column(self.pane, self._play_widget))

        if show:
            return template.servable().show()
        return template


def animation(*series, show=True, **kwargs):
    ani = Animation(*series, **kwargs)
    if show:
        return ani.show()
    return ani
