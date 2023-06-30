import os
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.series import *
from spb.backends.bokeh.renderers import *
from sympy.external import import_module
import warnings


class BokehBackend(Plot):
    """
    A backend for plotting SymPy's symbolic expressions using Bokeh.
    This implementation only supports 2D plots.

    Parameters
    ==========

    aspect : str
        Set the aspect ratio of a 2D plot. Default to ``None``. Set it to
        ``"equal"`` to sets equal spacing on the axis.

    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to Matplotlib's plot
        functions to customize the appearance of lines, surfaces, images,
        contours, quivers, streamlines...
        To learn more about customization:

        * Refer to:

          - [#fn1]_ to customize lines plots. Default to:
            ``dict(line_width = 2)``.
          - [#fn6]_ to customize scatter plots. Default to:
            ``dict(marker = "circle")``.

        * Default options for quiver plots:

          .. code-block:: python

           dict(
               scale = 1,
               pivot = "mid",      # "mid", "tip" or "tail"
               arrow_heads = True,  # show/hide arrow
               line_width = 1
           )

        * Default options for streamline plots:
          ``dict(line_width=2, line_alpha=0.8)``

    axis : boolean, optional
        Turns on/off the axis visibility (and associated tick labels).
        Default to True (axis are visible).

    theme : str, optional
        Set the theme. Find more Bokeh themes at [#fn2]_ .

    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the `bokeh.models.LabelSet` class.
        This feature is experimental. It might get removed in the future.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the `bokeh.models.Scatter` class.
        This feature is experimental. It might get removed in the future.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The ``"args"`` key must contain the
        `bokeh.models.ColumnDataSource` object containing the
        data. All other keyword arguments will be passed to the
        `bokeh.models.Rect` class.
        This feature is experimental. It might get removed in the future.

    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the `bokeh.models.VArea` class.
        This feature is experimental. It might get removed in the future.


    References
    ==========
    .. [#fn1] https://docs.bokeh.org/en/latest/docs/reference/plotting.html#bokeh.plotting.Figure.line
    .. [#fn2] https://docs.bokeh.org/en/latest/docs/reference/themes.html
    .. [#fn6] https://docs.bokeh.org/en/latest/docs/reference/plotting/figure.html#bokeh.plotting.Figure.scatter


    See also
    ========

    Plot, MatplotlibBackend, PlotlyBackend, K3DBackend
    """

    _library = "bokeh"
    _allowed_keys = Plot._allowed_keys + [
        "markers", "annotations", "fill", "rectangles"]

    colorloop = []
    colormaps = []
    cyclic_colormaps = []

    renderers_map = {
        LineOver1DRangeSeries: Line2DRenderer,
        List2DSeries: Line2DRenderer,
        Parametric2DLineSeries: Line2DRenderer,
        ColoredLineOver1DRangeSeries: Line2DRenderer,
        AbsArgLineSeries: Line2DRenderer,
        ComplexPointSeries: Line2DRenderer,
        Vector2DSeries: Vector2DRenderer,
        ComplexDomainColoringSeries: ComplexRenderer,
        ContourSeries: ContourRenderer,
        GeometrySeries: GeometryRenderer,
        GenericDataSeries: GenericRenderer,
        HVLineSeries: HVLineRenderer,
    }

    pole_line_kw = {"line_color": "#000000", "line_dash": "dotted"}

    def __init__(self, *args, **kwargs):
        self.np = import_module('numpy')
        self.bokeh = import_module(
            'bokeh',
            import_kwargs={'fromlist': ['models', 'events', 'plotting', 'io', 'palettes', 'embed', 'resources']},
            warn_not_installed=True,
            min_module_version='2.3.0')
        bp = self.bokeh.palettes
        cc = import_module(
            'colorcet',
            min_module_version='3.0.0')
        matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm']},
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        cm = matplotlib.cm

        self.colorloop = bp.Category10[10]
        self.colormaps = [cc.bmy, "aggrnyl", cc.kbc, cc.bjy, "plotly3"]
        self.cyclic_colormaps = [cm.hsv, cm.twilight, cc.cyclic_mygbm_30_95_c78_s25]

        self._init_cyclers()
        super().__init__(*args, **kwargs)

        if self.polar_axis:
            raise ValueError("BokehBackend doesn't support polar axis.")

        # set labels
        self._use_latex = kwargs.get("use_latex", cfg["bokeh"]["use_latex"])
        self._set_labels()
        self._set_title()

        self._theme = kwargs.get("theme", cfg["bokeh"]["theme"])

        self._run_in_notebook = False
        if self._get_mode() == 0:
            self._run_in_notebook = True
            self.bokeh.io.output_notebook(hide_banner=True)

        if ((len([s for s in self._series if s.is_2Dline]) > 10) and
            (not type(self).colorloop) and
            not ("process_piecewise" in kwargs.keys())):
            # add colors if needed
            self.colorloop = bp.Category20[20]

        self._handles = dict()

        # empty plots (len(series)==0) should only have x, y tooltips
        TOOLTIPS = [("x", "$x"), ("y", "$y")]
        if len(self.series) > 0:
            if all([s.is_parametric for s in self.series]):
                # with parametric plots, also visualize the parameter
                TOOLTIPS += [("u", "@us")]
            if any([s.is_complex and s.is_domain_coloring for s
                    in self.series]):
                # with complex domain coloring, shows the magnitude and phase
                # in the tooltip
                TOOLTIPS += [("Abs", "@abs"), ("Arg", "@arg")]

        sizing_mode = cfg["bokeh"]["sizing_mode"]
        # if any(s.is_complex and s.is_domain_coloring for s in self.series):
        #     # for complex domain coloring: the idea is to have the colorbar
        #     # closer to the actual plot, rather than having a lot of white
        #     # space in between.
        #     sizing_mode = None

        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        kw = dict(
            title=title,
            x_axis_label=xlabel if xlabel else "x",
            y_axis_label=ylabel if ylabel else "y",
            sizing_mode="fixed" if self.size else sizing_mode,
            width=int(self.size[0]) if self.size else cfg["bokeh"]["width"],
            height=int(self.size[1]) if self.size else cfg["bokeh"]["height"],
            x_axis_type=self.xscale,
            y_axis_type=self.yscale,
            tools="pan,wheel_zoom,box_zoom,reset,hover,save",
            tooltips=TOOLTIPS,
            match_aspect=True if self.aspect == "equal" else False,
        )
        if self.xlim:
            kw["x_range"] = self.xlim
        if self.ylim:
            kw["y_range"] = self.ylim
        self._fig = self.bokeh.plotting.figure(**kw)
        self._fig.axis.visible = self.axis
        self.grid = kwargs.get("grid", cfg["bokeh"]["grid"])
        self._fig.grid.visible = self.grid
        if cfg["bokeh"]["show_minor_grid"]:
            self._fig.grid.minor_grid_line_alpha = cfg["bokeh"]["minor_grid_line_alpha"]
            self._fig.grid.minor_grid_line_color = self._fig.grid.grid_line_color[0]
            self._fig.grid.minor_grid_line_dash = cfg["bokeh"]["minor_grid_line_dash"]
        if self._invert_x_axis:
            self._fig.x_range.flipped = True

        self._create_renderers()

    @property
    def fig(self):
        """Returns the figure."""
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            # if the backend was created without showing it
            self.draw()
        return self._fig

    def draw(self):
        """ Loop over data renderers, generates numerical data and add it to
        the figure. Note that this method doesn't show the plot.
        """
        self._process_renderers()

    process_series = draw

    def _set_piecewise_color(self, s, color):
        """Set the color to the given series"""
        if "color" not in s.rendering_kw:
            # only set the color if the user didn't do that already
            s.rendering_kw["color"] = color
            if s.is_point and (not s.is_filled):
                s.rendering_kw["fill_color"] = "white"

    @staticmethod
    def _do_sum_kwargs(p1, p2):
        kw = p1._copy_kwargs()
        kw["theme"] = p1._theme
        return kw

    def _process_renderers(self):
        self._init_cyclers()
        # clear figure. Must clear both the renderers as well as the
        # colorbars which are added to the right side.
        self._fig.renderers = []
        self._fig.right = []

        for r, s in zip(self.renderers, self.series):
            self._check_supported_series(r, s)
            r.draw()

        if len(self._fig.legend) > 0:
            # hide default legend
            self._fig.legend.visible = False
            # add a new legend only showing the appropriate items
            legend_items = []
            for s, r in zip(self.series, self._fig.renderers):
                if (s.show_in_legend and (s.is_2Dline or s.is_geometry) and
                    (not s.use_cm)):
                    legend_items.append(
                        self.bokeh.models.LegendItem(
                            label=s.get_label(self._use_latex), renderers=[r]))
            if self.legend and (len(legend_items) > 0):
                legend = self.bokeh.models.Legend(items=legend_items)
                # interactive legend
                legend.click_policy = "hide"
                self._fig.add_layout(legend, "right")

    def _get_img(self, img):
        np = import_module('numpy')
        new_img = np.zeros(img.shape[:2], dtype=np.uint32)
        pixel = new_img.view(dtype=np.uint8).reshape((*img.shape[:2], 4))
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                pixel[j, i] = [*img[j, i], 255]
        return new_img

    def _get_segments(self, x, y, u):
        # MultiLine works with line segments, not with line points! :|
        xs = [x[i - 1 : i + 1] for i in range(1, len(x))]
        ys = [y[i - 1 : i + 1] for i in range(1, len(y))]
        # let n be the number of points. Then, the number of segments
        # will be (n - 1). Therefore, we remove one parameter. If n is
        # sufficiently high, there shouldn't be any noticeable problem in
        # the visualization.
        us = u[:-1]
        return xs, ys, us

    def _create_gradient_line(self, x, y, u, colormap, name, line_kw, is_point=False):
        merge = self.merge
        if not is_point:
            xs, ys, us = self._get_segments(x, y, u)
        else:
            xs, ys, us = x, y, u
        color_mapper = self.bokeh.models.LinearColorMapper(
            palette=colormap, low=min(us), high=max(us))
        data_source = self.bokeh.models.ColumnDataSource(
            dict(xs=xs, ys=ys, us=us))

        lkw = dict(
            line_width=2,
            name=name,
            line_color={"field": "us", "transform": color_mapper},
        )
        kw = merge({}, lkw, line_kw)
        if not is_point:
            glyph = self.bokeh.models.MultiLine(xs="xs", ys="ys", **kw)
        else:
            glyph = self.bokeh.models.Scatter(x="xs", y="ys", **kw)
        colorbar = self.bokeh.models.ColorBar(
            color_mapper=color_mapper, title=name, width=8)
        return data_source, glyph, colorbar, kw

    def update_interactive(self, params):
        """Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()

        for r in self.renderers:
            if r.series.is_interactive:
                r.update(params)

        self._set_axes_texts()

    def _set_axes_texts(self):
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        self._fig.title = title
        self._fig.xaxis.axis_label = xlabel
        self._fig.yaxis.axis_label = ylabel

    def save(self, path, **kwargs):
        """ Export the plot to a static picture or to an interactive html file.

        Refer to [#fn3]_ and [#fn4]_ to visualize all the available keyword
        arguments.

        Notes
        =====

        1. In order to export static pictures, the user also need to install
           the packages listed in [#fn5]_.
        2. When exporting a fully portable html file, by default the necessary
           Javascript libraries will be loaded with a CDN. This creates the
           smallest file size possible, but it requires an internet connection
           in order to view/load the file and its dependencies.

        References
        ==========
        .. [#fn3] https://docs.bokeh.org/en/latest/docs/user_guide/export.html
        .. [#fn4] https://docs.bokeh.org/en/latest/docs/user_guide/embed.html
        .. [#fn5] https://docs.bokeh.org/en/latest/docs/reference/io.html#module-bokeh.io.export

        """
        merge = self.merge

        ext = os.path.splitext(path)[1]
        if ext.lower() in [".htm", ".html"]:
            CDN = self.bokeh.resources.CDN
            file_html = self.bokeh.embed.file_html
            skw = dict(resources=CDN, title="Bokeh Plot")
            html = file_html(self.fig, **merge(skw, kwargs))
            with open(path, 'w') as f:
                f.write(html)
        elif ext == ".svg":
            self._fig.output_backend = "svg"
            self.bokeh.io.export_svg(self.fig, filename=path)
        else:
            if ext == "":
                path += ".png"
            self._fig.output_backend = "canvas"
            self.bokeh.io.export_png(self._fig, filename=path)

    def show(self):
        """Visualize the plot on the screen."""
        if len(self._fig.renderers) != len(self.series):
            self._process_renderers()
        # if the backend it running from a python interpreter, the server
        # wont' work. Hence, launch a static figure, which doesn't listen
        # to events (no pan-auto-update).
        curdoc = self.bokeh.io.curdoc
        curdoc().theme = self._theme
        self.bokeh.plotting.show(self._fig)

    def _get_quivers_data(self, xs, ys, u, v, **quiver_kw):
        """Compute the segments coordinates to plot quivers.

        Parameters
        ==========
            xs : np.ndarray
                A 2D numpy array representing the discretization in the
                x-coordinate

            ys : np.ndarray
                A 2D numpy array representing the discretization in the
                y-coordinate

            u : np.ndarray
                A 2D numpy array representing the x-component of the vector

            v : np.ndarray
                A 2D numpy array representing the x-component of the vector

            kwargs : dict, optional
                An optional

        Returns
        =======
            data: dict
                A dictionary suitable to create a data source to be used with
                Bokeh's Segment.

            quiver_kw : dict
                A dictionary containing keywords to customize the appearance
                of Bokeh's Segment glyph
        """
        np = import_module('numpy')
        scale = quiver_kw.pop("scale", 1.0)
        pivot = quiver_kw.pop("pivot", "mid")
        arrow_heads = quiver_kw.pop("arrow_heads", True)

        xs, ys, u, v = [t.flatten() for t in [xs, ys, u, v]]

        magnitude = np.sqrt(u ** 2 + v ** 2)
        rads = np.arctan2(v, u)
        lens = magnitude / max(magnitude) * scale

        # Compute segments and arrowheads
        # Compute offset depending on pivot option
        xoffsets = np.cos(rads) * lens / 2.0
        yoffsets = np.sin(rads) * lens / 2.0
        if pivot == "mid":
            nxoff, pxoff = xoffsets, xoffsets
            nyoff, pyoff = yoffsets, yoffsets
        elif pivot == "tip":
            nxoff, pxoff = 0, xoffsets * 2
            nyoff, pyoff = 0, yoffsets * 2
        elif pivot == "tail":
            nxoff, pxoff = xoffsets * 2, 0
            nyoff, pyoff = yoffsets * 2, 0
        x0s, x1s = (xs + nxoff, xs - pxoff)
        y0s, y1s = (ys + nyoff, ys - pyoff)

        if arrow_heads:
            arrow_len = lens / 4.0
            xa1s = x0s - np.cos(rads + np.pi / 4) * arrow_len
            ya1s = y0s - np.sin(rads + np.pi / 4) * arrow_len
            xa2s = x0s - np.cos(rads - np.pi / 4) * arrow_len
            ya2s = y0s - np.sin(rads - np.pi / 4) * arrow_len
            x0s = np.tile(x0s, 3)
            x1s = np.concatenate([x1s, xa1s, xa2s])
            y0s = np.tile(y0s, 3)
            y1s = np.concatenate([y1s, ya1s, ya2s])

        data = {
            "x0": x0s,
            "x1": x1s,
            "y0": y0s,
            "y1": y1s,
        }

        return data, quiver_kw


BB = BokehBackend
