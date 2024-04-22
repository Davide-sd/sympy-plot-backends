import ipywidgets
from sympy.external import import_module
from spb.animation import BaseAnimation
from spb.defaults import TWO_D_B, THREE_D_B
from spb.interactive import IPlot
from spb import BB, MB, PlotGrid
from IPython.display import clear_output


class Animation(BaseAnimation, IPlot):
    def __init__(self, *series, **kwargs):
        plotgrid = kwargs.get("plotgrid", None)
        params = kwargs.get("params", {})
        self._original_params = params

        if plotgrid:
            self._backend = plotgrid
            self._post_init_plotgrid(**kwargs)
        else:
            is_3D = all([s.is_3D for s in series])
            Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
            kwargs["is_iplot"] = True
            kwargs["imodule"] = "ipywidgets"
            self._backend = Backend(*series, **kwargs)
            self._post_init_plot(**kwargs)

        play = ipywidgets.Play(
            value=0,
            min=0,
            max=self._animation_data.n_frames,
            step=1,
            interval=int(1000 / self._animation_data.fps),
            description="Press play",
            disabled=False
        )
        slider = ipywidgets.IntSlider(
            min=0,
            max=self._animation_data.n_frames,
            step=1
        )
        ipywidgets.jslink((play, 'value'), (slider, 'value'))
        play.observe(self._update, "value")
        self._play_widget = ipywidgets.HBox([play, slider])

    def _update(self, change):
        print("IPYAnimation._update")
        frame_idx = change["new"]
        self.update_animation(frame_idx)
        if isinstance(self._backend, BB):
            bokeh = import_module(
                'bokeh',
                import_kwargs={'fromlist': ['io']},
                warn_not_installed=True,
                min_module_version='2.3.0')
            with self._output_figure:
                clear_output(True) # NOTE: this is the cause of flickering
                bokeh.io.show(self._backend.fig)

    def _get_iplot_kw(self):
        return {
            "backend": type(self._backend)
        }

    def show(self):
        # create the output figure
        if (isinstance(self._backend, MB) or
            (isinstance(self._backend, PlotGrid) and self._backend.is_matplotlib_fig)):
            # without plt.ioff, picture will show up twice. Morover, there
            # won't be any update
            self._backend.plt.ioff()
            if isinstance(self._backend, PlotGrid):
                if not self._backend.imagegrid:
                    self._backend.fig.tight_layout()
            self._output_figure = ipywidgets.Box([self._backend.fig.canvas])
        elif isinstance(self._backend, BB):
            self._output_figure = ipywidgets.Output()
            bokeh = import_module(
                'bokeh',
                import_kwargs={'fromlist': ['io']},
                warn_not_installed=True,
                min_module_version='2.3.0')
            with self._output_figure:
                bokeh.io.show(self._backend.fig)
        else:
            self._output_figure = self._backend.fig

        if (isinstance(self._backend, MB) or
            (isinstance(self._backend, PlotGrid) and self._backend.is_matplotlib_fig)):
            # turn back interactive behavior with plt.ion, so that picture
            # will be updated.
            self._backend.plt.ion() # without it there won't be any update

        return ipywidgets.VBox([self._output_figure, self._play_widget])


def animation(*series, show=True, **kwargs):
    ani = Animation(*series, **kwargs)
    if show:
        return ani.show()
    return ani
