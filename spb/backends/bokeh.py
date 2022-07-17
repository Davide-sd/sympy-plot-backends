import os
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from sympy.external import import_module


def compute_streamlines(x, y, u, v, density=1.0):
    """Return streamlines of a vector flow.

    * x and y are 1d arrays defining an *evenly spaced* grid.
    * u and v are 2d arrays (shape [y,x]) giving velocities.
    * density controls the closeness of the streamlines.

    Credit: https://docs.bokeh.org/en/latest/docs/gallery/quiver.html
    """
    np = import_module('numpy')

    ## Set up some constants - size of the grid used.
    NGX = len(x)
    NGY = len(y)

    ## Constants used to convert between grid index coords and user coords.
    DX = x[1] - x[0]
    DY = y[1] - y[0]
    XOFF = x[0]
    YOFF = y[0]

    ## Now rescale velocity onto axes-coordinates
    u = u / (x[-1] - x[0])
    v = v / (y[-1] - y[0])
    speed = np.sqrt(u * u + v * v)
    ## s (path length) will now be in axes-coordinates, but we must
    ## rescale u for integrations.
    u *= NGX
    v *= NGY
    ## Now u and v in grid-coordinates.

    NBX = int(30 * density)
    NBY = int(30 * density)

    blank = np.zeros((NBY, NBX))

    bx_spacing = NGX / float(NBX - 1)
    by_spacing = NGY / float(NBY - 1)

    def blank_pos(xi, yi):
        return int((xi / bx_spacing) + 0.5), int((yi / by_spacing) + 0.5)

    def value_at(a, xi, yi):
        if type(xi) == np.ndarray:
            x = xi.astype(int)
            y = yi.astype(int)
        else:
            x = int(xi)
            y = int(yi)
        a00 = a[y, x]
        a01 = a[y, x + 1]
        a10 = a[y + 1, x]
        a11 = a[y + 1, x + 1]
        xt = xi - x
        yt = yi - y
        a0 = a00 * (1 - xt) + a01 * xt
        a1 = a10 * (1 - xt) + a11 * xt
        return a0 * (1 - yt) + a1 * yt

    def rk4_integrate(x0, y0):
        ## This function does RK4 forward and back trajectories from
        ## the initial conditions, with the odd 'blank array'
        ## termination conditions. TODO tidy the integration loops.

        def f(xi, yi):
            dt_ds = 1.0 / value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return ui * dt_ds, vi * dt_ds

        def g(xi, yi):
            dt_ds = 1.0 / value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return -ui * dt_ds, -vi * dt_ds

        check = lambda xi, yi: xi >= 0 and xi < NGX - 1 and yi >= 0 and yi < NGY - 1

        bx_changes = []
        by_changes = []

        ## Integrator function
        def rk4(x0, y0, f):
            ds = 0.01  # min(1./NGX, 1./NGY, 0.01)
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = blank_pos(xi, yi)
            xf_traj = []
            yf_traj = []
            while check(xi, yi):
                # Time step. First save the point.
                xf_traj.append(xi)
                yf_traj.append(yi)
                # Next, advance one using RK4
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + 0.5 * ds * k1x, yi + 0.5 * ds * k1y)
                    k3x, k3y = f(xi + 0.5 * ds * k2x, yi + 0.5 * ds * k2y)
                    k4x, k4y = f(xi + ds * k3x, yi + ds * k3y)
                except IndexError:
                    # Out of the domain on one of the intermediate steps
                    break
                xi += ds * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
                yi += ds * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
                # Final position might be out of the domain
                if not check(xi, yi):
                    break
                stotal += ds
                # Next, if s gets to thres, check blank.
                new_xb, new_yb = blank_pos(xi, yi)
                if new_xb != xb or new_yb != yb:
                    # New square, so check and colour. Quit if required.
                    if blank[new_yb, new_xb] == 0:
                        blank[new_yb, new_xb] = 1
                        bx_changes.append(new_xb)
                        by_changes.append(new_yb)
                        xb = new_xb
                        yb = new_yb
                    else:
                        break
                if stotal > 2:
                    break
            return stotal, xf_traj, yf_traj

        integrator = rk4

        sf, xf_traj, yf_traj = integrator(x0, y0, f)
        sb, xb_traj, yb_traj = integrator(x0, y0, g)
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]

        ## Tests to check length of traj. Remember, s in units of axes.
        if len(x_traj) < 1:
            return None
        if stotal > 0.2:
            initxb, inityb = blank_pos(x0, y0)
            blank[inityb, initxb] = 1
            return x_traj, y_traj
        else:
            for xb, yb in zip(bx_changes, by_changes):
                blank[yb, xb] = 0
            return None

    ## A quick function for integrating trajectories if blank==0.
    trajectories = []

    def traj(xb, yb):
        if xb < 0 or xb >= NBX or yb < 0 or yb >= NBY:
            return
        if blank[yb, xb] == 0:
            t = rk4_integrate(xb * bx_spacing, yb * by_spacing)
            if t is not None:
                trajectories.append(t)

    ## Now we build up the trajectory set. I've found it best to look
    ## for blank==0 along the edges first, and work inwards.
    for indent in range((max(NBX, NBY)) // 2):
        for xi in range(max(NBX, NBY) - 2 * indent):
            traj(xi + indent, indent)
            traj(xi + indent, NBY - 1 - indent)
            traj(indent, xi + indent)
            traj(NBX - 1 - indent, xi + indent)

    xs = [np.array(t[0]) * DX + XOFF for t in trajectories]
    ys = [np.array(t[1]) * DY + YOFF for t in trajectories]

    return xs, ys


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

        * Refer to [#fn1]_ to customize lines plots. Default to:
          ``dict(line_width = 2)``.

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

    theme : str, optional
        Set the theme. Default to ``"dark_minimal"``. Find more Bokeh themes
        at [#fn2]_ .

    update_event : bool, optional
        If True, the backend will update the data series over the visibile
        range whenever a pan-event is triggered. Default to True.


    References
    ==========
    .. [#fn1] https://docs.bokeh.org/en/latest/docs/reference/plotting.html?highlight=line#bokeh.plotting.Figure.line
    .. [#fn2] https://docs.bokeh.org/en/latest/docs/reference/themes.html


    See also
    ========

    Plot, MatplotlibBackend, PlotlyBackend, K3DBackend
    """

    _library = "bokeh"

    colorloop = []
    colormaps = []
    cyclic_colormaps = []

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.bokeh = import_module(
            'bokeh',
            import_kwargs={'fromlist': ['models', 'events', 'plotting', 'io', 'palettes', 'embed', 'resources']},
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

        # set labels
        self._use_latex = kwargs.get("use_latex", cfg["bokeh"]["use_latex"])
        self._set_labels()

        self._theme = kwargs.get("theme", cfg["bokeh"]["theme"])
        self._update_event = kwargs.get("update_event",
            cfg["bokeh"]["update_event"])

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
        if any(s.is_complex and s.is_domain_coloring for s in self.series):
            # for complex domain coloring
            sizing_mode = None

        self._fig = self.bokeh.plotting.figure(
            title=self.title,
            x_axis_label=self.xlabel if self.xlabel else "x",
            y_axis_label=self.ylabel if self.ylabel else "y",
            sizing_mode="fixed" if self.size else sizing_mode,
            width=int(self.size[0]) if self.size else 600,
            height=int(self.size[1]) if self.size else 400,
            x_axis_type=self.xscale,
            y_axis_type=self.yscale,
            x_range=self.xlim,
            y_range=self.ylim,
            tools="pan,wheel_zoom,box_zoom,reset,hover,save",
            tooltips=TOOLTIPS,
            match_aspect=True if self.aspect == "equal" else False,
        )
        self.grid = kwargs.get("grid", cfg["bokeh"]["grid"])
        self._fig.grid.visible = self.grid
        if cfg["bokeh"]["show_minor_grid"]:
            self._fig.grid.minor_grid_line_alpha = cfg["bokeh"]["minor_grid_line_alpha"]
            self._fig.grid.minor_grid_line_color = self._fig.grid.grid_line_color[0]
            self._fig.grid.minor_grid_line_dash = cfg["bokeh"]["minor_grid_line_dash"]
        self._fig.on_event(self.bokeh.events.PanEnd, self._pan_update)

    @property
    def fig(self):
        """Returns the figure."""
        if len(self.series) != len(self._fig.renderers):
            # if the backend was created without showing it
            self.process_series()
        return self._fig

    def process_series(self):
        """ Loop over data series, generates numerical data and add it to the
        figure.
        """
        self._process_series(self._series)

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

    def _process_series(self, series):
        np = import_module('numpy')
        merge = self.merge
        self._init_cyclers()
        # clear figure. Must clear both the renderers as well as the
        # colorbars which are added to the right side.
        self._fig.renderers = []
        self._fig.right = []

        for i, s in enumerate(series):
            kw = None

            if s.is_2Dline:
                if s.is_parametric and s.use_cm:
                    x, y, param = s.get_data()
                    colormap = (
                        next(self._cyccm)
                        if self._use_cyclic_cm(param, s.is_complex)
                        else next(self._cm)
                    )
                    ds, line, cb, kw = self._create_gradient_line(
                        x, y, param, colormap, s.get_label(self._use_latex), s.rendering_kw)
                    self._fig.add_glyph(ds, line)
                    if self.legend:
                        self._handles[i] = cb
                        self._fig.add_layout(cb, "right")
                else:
                    if s.is_parametric:
                        x, y, param = s.get_data()
                        source = {"xs": x, "ys": y, "us": param}
                    else:
                        x, y = s.get_data()
                        source = {
                            "xs": x if not s.is_polar else y * np.cos(x),
                            "ys": y if not s.is_polar else y * np.sin(x)
                        }

                    color = next(self._cl) if s.line_color is None else s.line_color
                    lkw = dict(line_width=2,
                        legend_label=s.get_label(self._use_latex),
                        color=color)
                    if not s.is_point:
                        kw = merge({}, lkw, s.rendering_kw)
                        self._fig.line("xs", "ys", source=source, **kw)
                    else:
                        lkw["size"] = 8
                        if not s.is_filled:
                            lkw["fill_color"] = "white"
                        kw = merge({}, lkw, s.rendering_kw)
                        self._fig.circle("xs", "ys", source=source, **kw)

            elif s.is_contour and (not s.is_complex):
                if s.is_polar:
                    raise NotImplementedError()
                x, y, z = s.get_data()
                x, y, zz = [t.flatten() for t in [x, y, z]]
                minx, miny, minz = min(x), min(y), min(zz)
                maxx, maxy, maxz = max(x), max(y), max(zz)

                cm = next(self._cm)
                ckw = dict(palette=cm)
                kw = merge({}, ckw, s.rendering_kw)

                self._fig.image(
                    image=[z],
                    x=minx,
                    y=miny,
                    dw=abs(maxx - minx),
                    dh=abs(maxy - miny),
                    **kw
                )

                colormapper = self.bokeh.models.LinearColorMapper(
                    palette=cm, low=minz, high=maxz)
                cbkw = dict(width=8, title=s.get_label(self._use_latex))
                colorbar = self.bokeh.models.ColorBar(
                    color_mapper=colormapper, **cbkw)
                self._fig.add_layout(colorbar, "right")
                self._handles[i] = colorbar

            elif s.is_2Dvector:
                if s.is_streamlines:
                    x, y, u, v = s.get_data()
                    sqk = dict(color=next(self._cl), line_width=2, line_alpha=0.8)
                    stream_kw = s.rendering_kw.copy()
                    density = stream_kw.pop("density", 2)
                    kw = merge({}, sqk, stream_kw)
                    xs, ys = compute_streamlines(
                        x[0, :], y[:, 0], u, v, density=density)
                    self._fig.multi_line(xs, ys, **kw)
                else:
                    x, y, u, v = s.get_data()
                    data, quiver_kw = self._get_quivers_data(x, y, u, v,
                        **s.rendering_kw.copy())
                    mag = data["magnitude"]

                    color_mapper = self.bokeh.models.LinearColorMapper(
                        palette=next(self._cm), low=min(mag), high=max(mag))
                    # don't use color map if a scalar field is visible or if
                    # use_cm=False
                    line_color = (
                        {"field": "magnitude", "transform": color_mapper}
                        if ((not s.use_quiver_solid_color) and s.use_cm)
                        else next(self._cl)
                    )
                    source = self.bokeh.models.ColumnDataSource(data=data)
                    qkw = dict(line_color=line_color, line_width=1, name=s.get_label(self._use_latex))
                    kw = merge({}, qkw, quiver_kw)
                    glyph = self.bokeh.models.Segment(
                        x0="x0", y0="y0", x1="x1", y1="y1", **kw)
                    self._fig.add_glyph(source, glyph)
                    if isinstance(line_color, dict):
                        colorbar = self.bokeh.models.ColorBar(
                            color_mapper=color_mapper, width=8, title=s.get_label(self._use_latex))
                        self._fig.add_layout(colorbar, "right")
                        self._handles[i] = colorbar

            elif s.is_complex and s.is_domain_coloring and not s.is_3Dsurface:
                x, y, mag, angle, img, colors = s.get_data()
                img = self._get_img(img)

                source = self.bokeh.models.ColumnDataSource(
                    {
                        "image": [img],
                        "abs": [mag],
                        "arg": [angle],
                    }
                )

                self._fig.image_rgba(
                    source=source,
                    x=x.min(),
                    y=y.min(),
                    dw=x.max() - x.min(),
                    dh=y.max() - y.min(),
                )

                if colors is not None:
                    # chroma/phase-colorbar
                    cm1 = self.bokeh.models.LinearColorMapper(
                        palette=[tuple(c) for c in colors],
                        low=-np.pi, high=np.pi)
                    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
                    labels = ["-π", "-π / 2", "0", "π / 2", "π"]
                    colorbar1 = self.bokeh.models.ColorBar(
                        color_mapper=cm1,
                        title="Argument",
                        ticker=self.bokeh.models.tickers.FixedTicker(ticks=ticks),
                        major_label_overrides={k: v for k, v in zip(ticks, labels)})
                    self._fig.add_layout(colorbar1, "right")

            elif s.is_geometry:
                x, y = s.get_data()
                color = next(self._cl)
                pkw = dict(alpha=0.5, line_width=2, line_color=color, fill_color=color)
                kw = merge({}, pkw, s.rendering_kw)
                self._fig.patch(x, y, **kw)

            else:
                raise NotImplementedError(
                    "{} is not supported by {}\n".format(type(s), type(self).__name__)
                    + "Bokeh only supports 2D plots."
                )

        if len(self._fig.legend) > 0:
            self._fig.legend.visible = self.legend
            # interactive legend
            self._fig.legend.click_policy = "hide"
            self._fig.add_layout(self._fig.legend[0], "right")

    def _pan_update(self):
        rend = self.fig.renderers
        for i, s in enumerate(self.series):
            if s.is_2Dline and not s.is_parametric:
                s.start = self._fig.x_range.start
                s.end = self._fig.x_range.end
                x, y = s.get_data()
                source = {"xs": x, "ys": y}
                rend[i].data_source.data.update(source)
            elif s.is_complex and s.is_2Dline and s.is_parametric and s.use_cm:
                # this is when absarg=True
                s.start = complex(self._fig.x_range.start)
                s.end = complex(self._fig.x_range.end)
                x, y, param = s.get_data()
                xs, ys, us = self._get_segments(x, y, param)
                rend[i].data_source.data.update({"xs": xs, "ys": ys, "us": us})
                if i in self._handles.keys():
                    cb = self._handles[i]
                    cb.color_mapper.update(low=min(us), high=max(us))

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

    def _create_gradient_line(self, x, y, u, colormap, name, line_kw):
        merge = self.merge
        xs, ys, us = self._get_segments(x, y, u)
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
        glyph = self.bokeh.models.MultiLine(xs="xs", ys="ys", **kw)
        colorbar = self.bokeh.models.ColorBar(
            color_mapper=color_mapper, title=name, width=8)
        return data_source, glyph, colorbar, kw

    def _update_interactive(self, params):
        np = import_module('numpy')

        rend = self.fig.renderers
        if len(rend) != len(self.series):
            self._process_series(self.series)

        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].params = params

                if s.is_2Dline and s.is_parametric and s.use_cm:
                    x, y, param = self.series[i].get_data()
                    xs, ys, us = self._get_segments(x, y, param)
                    rend[i].data_source.data.update({"xs": xs, "ys": ys, "us": us})
                    if i in self._handles.keys():
                        cb = self._handles[i]
                        cb.color_mapper.update(low=min(us), high=max(us))

                elif s.is_2Dline:
                    if s.is_parametric:
                        x, y, param = self.series[i].get_data()
                        source = {"xs": x, "ys": y, "us": param}
                    else:
                        x, y = self.series[i].get_data()
                        source = {
                            "xs": x if not s.is_polar else y * np.cos(x),
                            "ys": y if not s.is_polar else y * np.sin(x)
                        }
                    rend[i].data_source.data.update(source)

                elif s.is_contour and (not s.is_complex):
                    x, y, z = s.get_data()
                    cb = self._handles[i]
                    rend[i].data_source.data.update({"image": [z]})
                    zz = z.flatten()
                    # TODO: as of Bokeh 2.3.2, the following line is going to
                    # update the values of the color mapper, but the redraw
                    # is not applied, hence there is an error in the
                    # visualization. Keep track of the following issue:
                    # https://github.com/bokeh/bokeh/issues/11116
                    cb.color_mapper.update(low=min(zz), high=max(zz))

                elif s.is_2Dvector:
                    x, y, u, v = s.get_data()
                    if s.is_streamlines:
                        density = s.rendering_kw.copy().pop("density", 2)
                        xs, ys = compute_streamlines(
                            x[0, :], y[:, 0], u, v, density=density
                        )
                        rend[i].data_source.data.update({"xs": xs, "ys": ys})
                    else:
                        quiver_kw = s.rendering_kw.copy()
                        data, quiver_kw = self._get_quivers_data(
                            x, y, u, v, **quiver_kw
                        )
                        rend[i].data_source.data.update(data)

                        line_color = rend[i].glyph.line_color
                        if (not s.use_quiver_solid_color) and s.use_cm:
                            # update the colorbar
                            cmap = line_color["transform"].palette
                            mag = data["magnitude"]
                            color_mapper = self.bokeh.models.LinearColorMapper(
                                palette=cmap, low=min(mag),
                                high=max(mag))
                            line_color = quiver_kw.get(
                                "line_color",
                                {
                                    "field": "magnitude",
                                    "transform": color_mapper
                                },
                            )
                            rend[i].glyph.line_color = line_color

                elif s.is_complex and s.is_domain_coloring and not s.is_3Dsurface:
                    # TODO: for some unkown reason, domain_coloring and
                    # interactive plot don't like each other...
                    x, y, mag, angle, img, _ = s.get_data()
                    img = self._get_img(img)
                    source = {
                        "image": [img],
                        "abs": [mag],
                        "arg": [angle],
                    }
                    rend[i].data_source.data.update(source)

                elif s.is_geometry and (not s.is_2Dline):
                    x, y = s.get_data()
                    source = {"x": x, "y": y}
                    rend[i].data_source.data.update(source)

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
        if ext.lower() in [".html", ".html"]:
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

    def _launch_server(self, doc):
        """By launching a server application, we can use Python callbacks
        associated to events.
        """
        doc.theme = self._theme
        doc.add_root(self._fig)

    def show(self):
        """Visualize the plot on the screen."""
        if len(self._fig.renderers) != len(self.series):
            self._process_series(self._series)
        if self._run_in_notebook and self._update_event:
            # TODO: the current way we are launching the server only works
            # within Jupyter Notebook. Is there another way of launching
            # it so that it can run from any Python interpreter?
            self.bokeh.plotting.show(self._launch_server)
        else:
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
            "magnitude": np.tile(magnitude, 3),
        }

        return data, quiver_kw


BB = BokehBackend
