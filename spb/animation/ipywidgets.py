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
            max=self.animation_data.n_frames - 1,
            step=1,
            interval=int(1000 / self.animation_data.fps),
            description="Press play",
            disabled=False
        )
        slider = ipywidgets.IntSlider(
            min=0,
            max=self.animation_data.n_frames - 1,
            step=1
        )
        ipywidgets.jslink((play, 'value'), (slider, 'value'))
        play.observe(self._update, "value")
        self._play_widget = ipywidgets.HBox([play, slider])

    def _update(self, change):
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
    """Create an animation containing the plot and a few interactive controls
    (play/pause/loop buttons).

    Parameters
    ==========

    series : BaseSeries
        Instances of :py:class:`spb.series.BaseSeries`, representing the
        symbolic expression to be plotted.

    animation : bool or dict

        * ``False`` (default value): no animation.
        * ``True``: the animation will use the following default values:
          ``fps=30, time=5``.
        * ``dict``: the dictionary should contain these optional keys:

          - ``"fps"``: frames per second of the animation.
          - ``"time"``: total animation time.

        It must be noted that these values are exact only if the animation
        is going to be saved on a file. For interactive applications, these
        values are just indicative: the animation is going to compute new
        data at each time step. Hence, the more data needs to be computed,
        the slower the update.

    params : dict
        A dictionary mapping the symbols to a parameter. The parameter can be:

        1. a tuple of the form `(min, max, spacing)`, where:

           - min, max : float
                Minimum and maximum values. Must be finite numbers.
           - spacing : str, optional
                Specify the discretization spacing. Default to ``"linear"``,
                can be changed to ``"log"``.

           This can be used to simulate a slider.

        2. A dictionary, mapping specific animation times to parameter values.
           This is useful to simulate steps (or values from a dropdown widget,
           or a spinner widget).
           For example, let ``tf`` be the animation time. Then, this
           dictionary, ``{t1: v1, t2: v2}``, creates the following steps:

           - 0 for `0 <= t <= t1`.
           - v1 for `t1 <= t <= t2`.
           - v2 for `t2 <= t <= tf`.

        3. A 1D numpy array, with length given by ``fps * time``, specifying
           the custom value at each animation frame (or, at each time step)
           associated to the parameter.

    show : bool, optional
        Default to True.
        If True, it will return an object that will be rendered on the
        output cell of a Jupyter Notebook. If False, it returns an instance
        of ``Animation``, which can later be shown by calling the
        ``show()`` method, or saved to a GIF/MP4 file using the
        ``save()`` method.

    title : str or tuple
        The title to be shown on top of the figure. To specify a parametric
        title, write a tuple of the form:``(title_str, param_symbol1, ...)``,
        where:

        * ``title_str`` must be a formatted string, for example:
          ``"test = {:.2f}"``.
        * ``param_symbol1, ...`` must be a symbol or a symbolic expression
          whose free symbols are contained in the ``params`` dictionary.

    Notes
    =====

    1. Animations are a special kind of interactive applications.
       In particular, this function can only be shown on Jupyter Notebook
       because it uses
       :py:class:`ipywidgets.widgets.widget_int.Play`.
    2. Animations with Matplotlib requires the ``%matplotlib widget`` command
       to be executed at the top of the Jupyter Notebook. This magic command
       comes from the
       `ipympl module <https://github.com/matplotlib/ipympl>`_ .
    3. Animations can be exported to GIF files or MP4 videos by calling the
       ``save()`` method. More on this in the examples.
    4. Saving K3D-Jupyter animations is particularly slow.

    Examples
    ========

    NOTE: the following examples use the ordinary plotting functions because
    ``animation`` is already integrated with them.

    Simple animation with two parameters. Note that:

    * the first parameter goes from a maximum value to a minimum value.
    * the second parameter goes from a minimum value to a maximum value.
    * a parametric title has been set.
    * a set of buttons allow to control the playback of the animation.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy import *
       from spb import *
       a, b, x = symbols("a b x")
       max_r = 30
       plot_parametric(
           x * cos(x), x * sin(x), prange(x, a, b),
           aspect="equal", use_cm=False,
           animation=True, params={a: (10, 0), b: (20, max_r)},
           title=("a={:.2f}; b={:.2f}", a, b),
           xlim=(-max_r, max_r), ylim=(-max_r, max_r)
       )

    Animation showing how to:

    * set the frames-per-second and total animation time.
    * set custom values to a parameter. Here, ``b`` will vary from 0 to 1 and
      back to 0 following a sinusoidal law.
    * set a parametric title.

    .. code-block::

       from sympy import *
       from spb import *
       import numpy as np
       a, b, x = symbols("a b x")
       fps = 20
       time = 6
       frames = np.linspace(0, 2*np.pi, fps * time)
       b_values = (np.sin(frames - np.pi/2) + 1) / 2
       plot(
           cos(a * x) * exp(-abs(x) * b), (x, -pi, pi),
           params={
               a: (0.5, 3),
               b: b_values
           },
           animation={"fps": fps, "time": time},
           title=("a={:.2f}, b={:.2f}", a, b),
           ylim=(-1.25, 1.25)
       )

    .. video:: _static/animations/matplotlib-animation.mp4
       :width: 600

    Animation of a 3D surface using K3D-Jupyter. Here we create an
    ``Animation`` object, which can later be used to save the animation
    to a file.

    .. code-block::

       from sympy import *
       from spb import *
       import numpy as np
       r, theta, t, a = symbols("r, theta, t, a")
       expr = cos(r**2 - a) * exp(-r / 3)
       p = plot_surface_revolution(
           expr, (r, 0, 5), (theta, 0, t),
           params={t: (1e-03, 2*pi), a: (0, 2*pi)},
           use_cm=True, color_func=lambda x, y, z: np.sqrt(x**2 + y**2),
           is_polar=True,
           wireframe=True, wf_n1=30, wf_n2=30,
           wf_rendering_kw={"width": 0.005},
           animation=True,
           title=(r"theta={:.4f}; \, a={:.4f}", t, a),
           backend=KB, grid=False, show=False
       )
       p.show()
       # Use the mouse to properly orient the view and then save the animation
       p.save("3d-animation.mp4")

    .. video:: _static/animations/3d_animation.mp4
       :width: 600

    Evolution of a complex function using the graphics module and Plotly.
    Note that ``animation=True`` has been set in the ``graphics()``
    function call.

    .. code-block::

       from sympy import *
       from spb import *
       t, tf = symbols("t t_f", real=True)
       gamma, omega = Rational(1, 2), 7
       f = exp(-gamma * t**2) * exp(I * omega * t)
       params = {tf: (-5, 5)}
       graphics(
           line_parametric_3d(
               t, re(f), im(f), range=(t, -5, tf), label="f(t)",
               params=params, use_cm=False
           ),
           line_parametric_3d(
               t, re(f), -2, range=(t, -5, tf), label="re(f(t))",
               params=params, use_cm=False
           ),
           line_parametric_3d(
               t, 2, im(f), range=(t, -5, tf), label="im(f(t))",
               params=params, use_cm=False
           ),
           line_parametric_3d(
               5, re(f), im(f), range=(t, -5, tf), label="abs(f(t))",
               params=params, use_cm=False
           ),
           backend=PB, aspect=dict(x=3, y=1, z=1), ylim=(-2, 2), zlim=(-2, 2),
           xlabel="t", ylabel="re(f)", zlabel="im(f)",
           title="$f(t)=%s$" % latex(f), animation=True, size=(704, 512)
       )

    .. video:: _static/animations/graphics-animation.mp4
       :width: 600

    Plotgrid animation. Note that:

    * Each plot is an interactive animation (they can also be ordinary plots).
    * ``p2, p3`` use defaults fps/time, whereas ``p1`` uses custom values.
    * the overall plotgrid parses the different animations, collecting times
      and fps. It then choses the highest numbers, and recreates all the
      parameters. Hence, in the following animation ``p1`` runs for the entire
      animation.

    .. code-block::

       from sympy import *
       from spb import *
       from matplotlib.gridspec import GridSpec
       a, b, c, d, x, y, z = symbols("a:d x:z")
       p1 = plot(
           sin(a*x), cos(a*x),
           animation={"fps": 10, "time": 2}, params={a: (1, 5)}, show=False)
       max_r = 30
       p2 = plot_parametric(
           x * cos(x), x * sin(x), prange(x, b, c),
           aspect="equal", use_cm=False,
           animation=True, params={b: (10, 0), c: (20, max_r)},
           title=("b={:.2f}; c={:.2f}", b, c),
           show=False, xlim=(-max_r, max_r), ylim=(-max_r, max_r))
       p3 = plot_complex(gamma(d*z), (z, -3-3*I, 3+3*I), title=r"$\gamma(d \, z)$",
           animation=True, params={d: (-1, 1)}, coloring="b", grid=False, show=False)
       gs = GridSpec(3, 4)
       mapping = {
           gs[2, :]: p1,
           gs[0:2, 0:2]: p2,
           gs[0:2, 2:]: p3,
       }
       plotgrid(gs=mapping)

    .. video:: _static/animations/plotgrid-animation.mp4
       :width: 600

    """

    ani = Animation(*series, **kwargs)
    if show:
        return ani.show()
    return ani
