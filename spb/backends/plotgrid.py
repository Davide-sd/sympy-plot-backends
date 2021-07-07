from spb.backends.base_backend import Plot
from spb.backends.matplotlib import MB
import matplotlib.pyplot as plt
import numpy as np
import panel as pn

def _nrows_ncols(nr, nc, nplots):
    """ Based on the number of plots to be shown, define the correct number of
    rows and/or columns.
    """
    if (nc <= 0) and (nr <= 0):
        print("a")
        nc = 1
        nr = nplots
    elif nr <= 0:
        print("b")
        nr = int(np.ceil(nplots / nc))
    elif nc <= 0:
        print("c")
        nc = int(np.ceil(nplots / nr))
    elif nr * nc < nplots:
        print("d")
        nr = int(np.ceil(nr / nc))
    return nr, nc

def plotgrid(*args, **kwargs):
    """ Combine multiple plots into a grid-like layout.
    This function has two modes of operation, depending on the input arguments.
    Make sure to read the examples to fully understand them.

    Parameters
    ==========
        args : sequence
            A sequence of aldready created plots. This, in combination with
            `nr` and `nc` represents the first mode of operation, where a basic 
            grid with (nc * nr) subplots will be created.
        
    Keyword Arguments
    =================
        nr, nc : int
            Number of rows and columns.
            By default, `nc = 1` and `nr = -1`: this will create as many rows as
            necessary, and a single column.
            If we set `nc = 1` and `nc = -1`, it will create as many column as
            necessary, and a single row.
        
        gs : dict
            A dictionary mapping Matplotlib's GridSpect objects to the plots.
            The keys represent the cells of the layout. Each cell will host the
            associated plot.
            This represents the second mode of operation, as it allows to create
            more complicated layouts.
            NOTE: all plots must be instances of MatplotlibBackend!
    
    Returns
    =======
        Depending on the types of plots, this function returns either:
        * None: if all plots are instances of MatplotlibBackend.
        * an instance of holoviz's panel GridSpec, which will be rendered on
        Jupyter Notebook when mixed types of plots are received or when all the
        plots are not instances of MatplotlibBackend. Read the following
        documentation page to get more information:
        https://panel.holoviz.org/reference/layouts/GridSpec.html
    

    Examples
    ========

    First mode of operation with MatplotlibBackends:

    .. code-block:: python
        from sympy import *
        from spb.backends.matplotlib import MB
        from spb import *

        x, y, z = symbols("x, y, z")
        p1 = plot(sin(x), backend=MB, show=False)
        p2 = plot(tan(x), backend=MB, detect_poles=True, show=False)
        p3 = plot(exp(-x), backend=MB, show=False)
        plotgrid(p1, p2, p3)
    
    First mode of operation with different backends. Try this on a Jupyter
    Notebook. Note that Matplotlib as been integrated as a picture, thus it
    loses its interactivity.

    .. code-block:: python
        p1 = plot(sin(x), backend=MB, show=False)
        p2 = plot(tan(x), backend=MB, detect_poles=True, show=False)
        p3 = plot(exp(-x), backend=MB, show=False)
        plotgrid(p1, p2, p3, nr=1, nc=3)

    Second mode of operation: using Matplotlib GridSpec and all plots are
    instances of MatplotlibBackend:

    .. code-block:: python
        from matplotlib.gridspec import GridSpec
        
        p1 = plot(sin(x), cos(x), show=False, backend=MB)
        p2 = plot_contour(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), show=False, backend=BB)
        p3 = complex_plot(sqrt(x), show=False, backend=PB)
        p4 = vector_plot(Matrix([-y, x]), (x, -5, 5), (y, -5, 5), show=False, backend=MB)
        p5 = complex_plot(gamma(z), (z, -3-3*I, 3+3*I), show=False, backend=MB)

        gs = GridSpec(3, 3)
        mapping = {
            gs[0, :1]: p1,
            gs[1, :1]: p2,
            gs[2:, :1]: p3,
            gs[2:, 1:]: p4,
            gs[0:2, 1:]: p5,
        }
        plotgrid(gs=mapping)

    """
    gs = kwargs.get("gs", None)
        
    if gs is None:
        # default layout: 1 columns, as many rows as needed
        nr = kwargs.get("nr", -1)
        nc = kwargs.get("nc", 1)
        nr, nc = _nrows_ncols(nr, nc, len(args))
            
        if all(isinstance(a, MB) for a in args):
            fig, ax = plt.subplots(nr, nc)
            ax = ax.flatten()

            c = 0
            for i in range(nr):
                for j in range(nc):
                    if c < len(args):
                        kw = args[c]._kwargs
                        kw["backend"] = MB
                        kw["fig"] = fig
                        kw["ax"] = ax[c]
                        p = Plot(*args[c].series, **kw)
                        p.process_series()
                        c += 1
        else:
            fig = pn.GridSpec(sizing_mode='stretch_width')
            c = 0
            for i in range(nr):
                for j in range(nc):
                    if c < len(args):
                        p = args[c]
                        c += 1
                        if isinstance(p, MB) and (not hasattr(p, "ax")):
                            # if the MatplotlibBackend was created without
                            # showing it
                            p.process_series()
                        fig[i, j] = p.fig if not isinstance(p, MB) else p.fig[0]

    else:
        fig = plt.figure()
        axes = dict()
        for gs, p in gs.items():
            axes[fig.add_subplot(gs)] = p
            
        for ax, p in axes.items():
            kw = p._kwargs
            kw["backend"] = MB
            kw["fig"] = fig
            kw["ax"] = ax
            newplot = Plot(*p.series, **kw)
            newplot.process_series()
    
    if isinstance(fig, plt.Figure):
        fig.tight_layout()
        fig.show()
    else:
        return fig
