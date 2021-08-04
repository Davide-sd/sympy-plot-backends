3 - Customize The Module
------------------------

Let's suppose we have identified two backends that we like (one for 2D plots,
the other for 3D plots). Then, instead of passing in the keyword
``backend=SOMETHING`` each time we need to create a plot, we can customize the
module and set the default backends.

Let's import the necessary tools:

.. code-block:: python

    from spb.defaults import cfg, set_defaults
    display(cfg)


.. parsed-literal::

    {'plotly': {'theme': 'plotly_dark'},
     'bokeh': {'theme': 'dark_minimal', 'sizing_mode': 'stretch_width'},
     'k3d': {'bg_color': 3620427, 'grid_color': 8947848, 'label_color': 14540253},
     'matplotlib': {'axis_center': None,
        'grid': True,
        'use_jupyterthemes': False,
        'jupytertheme': None},
     'backend_2D': 'plotly',
     'backend_3D': 'k3d',
     'complex': {'modules': None, 'coloring': 'a'}}

And visualize the documentation:

.. code:: ipython3

    help(set_defaults)


.. parsed-literal::

    Help on function set_defaults in module spb.defaults:
    
    set_defaults(cfg)
        Set the default options for the plotting backends.
        
        Parameters
        ==========
            cfg : dict
                Dictionary containing the new values
        
        Examples
        ========
        
        Change the default 2D plotting backend to MatplotlibBackend.
        
            >>> from spb.defaults import cfg, set_defaults
            >>> ## to visualize the current settings
            >>> # print(cfg)
            >>> cfg["backend_2D"] = "matplotlib"
            >>> set_defaults(cfg)
    


We need to change the values in the ``cfg`` dictionary and then use the
``set_defaults`` function to apply the new configuration.

Let's say we would like to:

* use Bokeh for 2D plots and Plotly for 3D plots;
* use ``"seaborn"`` theme with Plotly.

Then:


.. code:: ipython3

    # we write the name of the plotting library
    # available options: bokeh, matplotlib, k3d, plotly
    cfg["backend_2D"] = "bokeh"
    cfg["backend_3D"] = "plotly"
    # the following depends on the plotting library
    cfg["plotly"]["theme"] = "seaborn"
    set_defaults(cfg)

Then, we restart the kernel and launch a couple of tests:

.. code:: ipython3

    from sympy import *
    from spb import *
    var("u, v, x, y")
    plot(sin(x), cos(x), log(x))

.. raw:: html
	:file: figs/tut-3/fig-01.html

.. code:: ipython3

    n = 400
    r = 2 + sin(7 * u + 5 * v)
    expr = (
        r * cos(u) * sin(v),
        r * sin(u) * sin(v),
        r * cos(v)
    )
    plot3d_parametric_surface(*expr, (u, 0, 2 * pi), (v, 0, pi), n=n)

.. raw:: html
	:file: figs/tut-3/fig-02.html

Let's now discuss a few customization options. The user can read the
documentation of each backend to find out more information.

.. code:: python

   # Set the location of the intersection between the horizontal and vertical
   # axis of Matplotlib (only works for 2D plots). Possible values:
   #       "center", "auto" or None
   # If None, use a standard Matplotlib layout with vertical axis on the left,
   # horizontal axis on the bottom.
   cfg["matplotlib"]["axis_center"] = None

   # Turn on the grid on Matplotlib plots
   cfg["matplotlib"]["grid"] = True

   # Find more Plotly themes at the following page:
   # https://plotly.com/python/templates/
   cfg["plotly"]["theme"] = "plotly_dark"

   # Find more Bokeh themes at the following page:
   # https://docs.bokeh.org/en/latest/docs/reference/themes.html
   cfg["bokeh"]["theme"] = "dark_minimal"

   # K3D-Jupyter colors are represented by an integer number.
   # For example, to set a white background:
   cfg["k3d"]["bg_color"] = 0xffffff

   # we can set the numerical evaluation library for complex plots.
   # Available options: "mpmath" or None (the latter uses Numpy/Scipy)
   cfg["complex"] = "mpmath"
   
   # set a default (complex) domain coloring option.
   cfg["coloring"] = "b"


Remember: every time we change a custom option, it is recommended to
restart the kernel in order to make changes effective.
