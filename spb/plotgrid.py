from sympy.external import import_module
from spb.backends.base_backend import Plot
from spb.backends.matplotlib import MB
from spb.backends.plotly import PB


def _nrows_ncols(nr, nc, nplots):
    """Define the correct number of rows and/or columns based on the number
    of plots to be shown.
    """
    np = import_module('numpy')

    if (nc <= 0) and (nr <= 0):
        nc = 1
        nr = nplots
    elif nr <= 0:
        nr = int(np.ceil(nplots / nc))
    elif nc <= 0:
        nc = int(np.ceil(nplots / nr))
    elif nr * nc < nplots:
        nr += 1
        return _nrows_ncols(nr, nc, nplots)
    return nr, nc


def _create_mpl_figure(mapping, imagegrid=False, size=None):
    matplotlib = import_module(
        'matplotlib',
        import_kwargs={'fromlist': ['pyplot', 'gridspec']},
        min_module_version='1.1.0',
        catch=(RuntimeError,))
    mpl_toolkits = import_module(
        'mpl_toolkits',
        import_kwargs={'fromlist': ['axes_grid1']},
        catch=(RuntimeError,))
    plt = matplotlib.pyplot

    print("size", size)
    kw = {} if not size else {"figsize": size}
    fig = plt.figure(**kw)

    if imagegrid:
        gs =list(mapping.keys())[0].get_gridspec()
        grid = mpl_toolkits.axes_grid1.ImageGrid(
            fig, 111,
            nrows_ncols=(gs.nrows, gs.ncols),
            axes_pad=0.15,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="7%",
            cbar_pad=0.15,
        )
        for (_, p), ax in zip(mapping.items(), grid):
            # cpa: current plot attributes
            cpa = p._copy_kwargs()
            cpa["backend"] = MB
            cpa["fig"] = fig
            cpa["ax"] = ax
            cpa["imagegrid"] = True
            p = Plot(*p.series, **cpa)
            p.process_series()
        return fig

    for spec, p in mapping.items():
        kw = {"projection": "3d"} if (len(p.series) > 0 and
            p.series[0].is_3D) else ({"projection": "polar"} if p.polar_axis
            else {})
        cur_ax = fig.add_subplot(spec, **kw)
        # cpa: current plot attributes
        cpa = p._copy_kwargs()
        cpa["backend"] = MB
        cpa["fig"] = fig
        cpa["ax"] = cur_ax
        p = Plot(*p.series, **cpa)
        p.process_series()
    return fig


def _create_panel_figure(mapping, panel_kw):
    pn = import_module(
        'panel',
        min_module_version='0.12.0')

    pn.extension("plotly")

    fig = pn.GridSpec(**panel_kw)
    for spec, p in mapping.items():
        rs = spec.rowspan
        cs = spec.colspan
        if isinstance(p, PB):
            d = {"data": list(p.fig.data), "layout": p.fig.layout}
            fig[slice(rs.start, rs.stop), slice(cs.start, cs.stop)] = pn.pane.Plotly(d)
        else:
            fig[slice(rs.start, rs.stop), slice(cs.start, cs.stop)] = pn.pane.panel(p.fig)
    return fig


def plotgrid(*args, **kwargs):
    """Combine multiple plots into a grid-like layout.
    This function has two modes of operation, depending on the input arguments.
    Make sure to read the examples to fully understand them.

    Parameters
    ==========

    args : sequence (optional)
        A sequence of aldready created plots. This, in combination with
        `nr` and `nc` represents the first mode of operation, where a basic
        grid with (nc * nr) subplots will be created.

    nr, nc : int (optional)
        Number of rows and columns.
        By default, `nc = 1` and `nr = -1`: this will create as many rows
        as necessary, and a single column.
        By setting `nr = 1` and `nc = -1`, it will create a single row and
        as many columns as necessary.

    gs : dict (optional)
        A dictionary mapping Matplotlib's `GridSpec` objects to the plots.
        The keys represent the cells of the layout. Each cell will host the
        associated plot.
        This represents the second mode of operation, as it allows to create
        more complicated layouts.

    panel_kw : dict (optional)
        A dictionary of keyword arguments to be passed to panel's `GridSpec`
        for further customization. Default to
        `dict(sizing_mode="stretch_width")`. Refer to [#fn1]_ for more
        information.

    show : boolean (optional)
        It applies only to Matplotlib figures. Default to True.

    Returns
    =======

    fig : `plt.Figure` or `pn.GridSpec`
        If all input plots are instances of `MatplotlibBackend`, than a
        Matplotlib `Figure` will be returned. Otherwise an instance of
        Holoviz Panel's `GridSpec` will be returned.


    Examples
    ========

    First mode of operation with instances of `MatplotlibBackend`:

    .. plot::
       :include-source: True
       :context: reset

       from sympy import symbols, sin, cos, tan, exp, sqrt, Matrix, gamma, I
       from spb import *

       x, y, z = symbols("x, y, z")
       p1 = plot(sin(x), backend=MB, title="sin(x)", show=False)
       p2 = plot(tan(x), backend=MB, adaptive=False, detect_poles=True,
            title="tan(x)", show=False)
       p3 = plot(exp(-x), backend=MB, title="exp(-x)", show=False)
       fig = plotgrid(p1, p2, p3)

    Second mode of operation, using Matplotlib GridSpec:

    .. plot::
       :include-source: True
       :context: reset

       from sympy import *
       from spb import *
       from matplotlib.gridspec import GridSpec
       x, y, z = symbols("x, y, z")
       p1 = plot(sin(x), cos(x), adaptive=False, show=False)
       expr = Tuple(1, sin(x**2 + y**2))
       p2 = plot_vector(expr, (x, -2, 2), (y, -2, 2),
            streamlines=True, scalar=False, use_cm=False,
            title=r"$\\vec{F}(x, y) = %s$" % latex(expr),
            xlabel="x", ylabel="y", show=False)
       p3 = plot_complex(gamma(z), (z, -3-3*I, 3+3*I), title=r"$\gamma(z)$",
            grid=False, show=False)

       gs = GridSpec(3, 4)
       mapping = {
           gs[2, :]: p1,
           gs[0:2, 0:2]: p2,
           gs[0:2, 2:]: p3,
       }
       fig = plotgrid(gs=mapping)

    References
    ==========

    .. [#fn1] https://panel.holoviz.org/reference/layouts/GridSpec.html

    """
    matplotlib = import_module(
        'matplotlib',
        import_kwargs={'fromlist': ['pyplot', 'gridspec']},
        min_module_version='1.1.0',
        catch=(RuntimeError,))
    plt = matplotlib.pyplot
    GridSpec = matplotlib.gridspec.GridSpec

    show = kwargs.get("show", True)
    gs = kwargs.get("gs", None)
    panel_kw = kwargs.get("panel_kw", dict(sizing_mode="stretch_width"))
    imagegrid = kwargs.get("imagegrid", False)
    size = kwargs.get("size", None)

    if (gs is None) and (len(args) == 0):
        fig = plt.figure()

    elif (gs is None):
        ### First mode of operation
        # default layout: 1 columns, as many rows as needed
        nr = kwargs.get("nr", -1)
        nc = kwargs.get("nc", 1)
        nr, nc = _nrows_ncols(nr, nc, len(args))

        gs = GridSpec(nr, nc)
        mapping = {}
        c = 0
        for i in range(nr):
            for j in range(nc):
                if c < len(args):
                    mapping[gs[i, j]] = args[c]
                c += 1

        if all(isinstance(a, MB) for a in args):
            fig = _create_mpl_figure(mapping, imagegrid, size)
        else:
            fig = _create_panel_figure(mapping, panel_kw)

    else:
        ### Second mode of operation
        if not isinstance(gs, dict):
            raise TypeError("`gs` must be a dictionary.")

        SubplotSpec = matplotlib.gridspec.SubplotSpec
        if not isinstance(list(gs.keys())[0], SubplotSpec):
            raise ValueError(
                "Keys of `gs` must be of elements of type "
                "matplotlib.gridspec.SubplotSpec. Use "
                "matplotlib.gridspec.GridSpec to create them.")

        if all(isinstance(a, MB) for a in gs.values()):
            fig = _create_mpl_figure(gs, imagegrid, size)
        else:
            fig = _create_panel_figure(gs, panel_kw)

    if isinstance(fig, plt.Figure):
        if not imagegrid:
            fig.tight_layout()
        if show:
            plt.show()
    return fig
