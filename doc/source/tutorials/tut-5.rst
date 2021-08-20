5 - Advanced Parametric-Interactive Plots
-----------------------------------------

There are times where ``iplot`` is not sufficient for our needs. Maybe we need
to use specific plot commands not included in the basic backends. Maybe we
needs to create more plots connected to the same parameters. Maybe we need
to use more complex widgets.

In such cases we will have to use
`holoviz's panel <https://panel.holoviz.org/index.html>`_. It is impossible to
learn everything ``panel`` has to offer with a couple of examples, therefore
the Reader is encouraged to explore the project website for demos and tutorials.

The following examples will focus on SymPy, but the same technique can be used
with any other scientific library.

For the purpose of these tutorials, we are going to use Bokeh as the plotting
library.


.. code-block:: python

   from sympy import *
   init_printing(use_latex=True)
   from spb import get_plot_data, plot
   from spb.backends.bokeh import BB
   import numpy as np
   import panel as pn
   pn.extension()


Example 1
=========

Let's suppose we have just computed a symbolic transfer function. We'd like
to plot the Bode and Nyquist plots.

.. code-block:: python

   kp, t, z, o = symbols("k_P, tau, zeta, omega")
   G = kp / (I**2 * t**2 * o**2 + 2 * z * t * o * I + 1)
   G

``o`` (omega, the pulsation) is the discretization variable. The parameters are:

* ``kp``: proportional gain;
* ``t`` (tau): response time;
* ``z`` (zeta): damping coefficient.

First, we are going to define the widgets that will compose the GUI. For a
list of available widgets consult
`the following documentation page <https://panel.holoviz.org/user_guide/Widgets.html#types-of-widgets>`_.

In particular, we'd like three float sliders for the aforementioned parameters,
and a button to switch from the Bode plot to Nyquist plot. To quickly create
the sliders, we will use the ``create_widgets`` function, which let us use a
syntax similar to the definition of parameters in the ``iplot`` function.

.. code-block:: python

   from spb.interactive import create_widgets

   sliders = create_widgets({
       kp: (1, 0, 3, 50, "Gain"),
       t: (1, 0, 3, 50, "Response Time"),
       z: (0.2, 0, 1, 50, "Damping Coefficient")
   })

   # use pn.widgets to create more complex widgets or for maximum customization
   plot_type = pn.widgets.RadioButtonGroup(
       name="Plot Type", options=["Bode", "Nyquist"], button_type='success')

   # create empty figures
   fig1 = plot(xlabel="[rad / s]", ylabel="Amplitude [dB]", xscale="log",
               backend=BB, show=False).fig
   fig2 = plot(xlabel="[rad / s]", ylabel="Phase [rad]", xscale="log",
               backend=BB, show=False).fig
   fig3 = plot(xlabel="Re(G)", ylabel="Im(G)", backend=BB, show=False).fig

Note that figures are also widgets. Here, we used the ``plot`` function to
quickly initialize the figures (note the keyword argument ``show=False``).

Now, let's create an interactive data series that will be later used to
generate the numerical data. To do so, we call the ``get_plot_data`` function,
providing a ``params`` dictionary mapping the parameters to numerical values,
and set ``get_series=True``. If we set ``get_series=False`` (which is the
default value), then numerical data will be returned instead of the data series.

.. code-block:: python

   s = get_plot_data(G, (o, 1e-03, 1e02), params={
       kp: sliders[kp].value,
       t: sliders[t].value,
       z: sliders[z].value,
   }, n=1000, xscale="log", get_series=True)

Since Bode plots uses a logarithm x-axis, we also specified ``xscale="log"``
to use a logarithm spacing in the discretization points.

Now, we need to create a function that will be called whenever we move the
sliders. This function will either add or update the data on the figures.

.. code-block:: python

   @pn.depends(sliders[kp], sliders[t], sliders[z], plot_type)
   def update(kpval, tval, zval, ptval):
       # this step is mandatory: it informs the series of the
       # availability of new parameters
       s.update_data({
           kp: kpval,
           t: tval,
           z: zval,
       })
       x, y = s.get_data()
        
       if ptval == "Bode":
           if len(fig1.renderers) == 0:
               fig1.line(x, 20 * np.log10(abs(y)), line_width=2)
               fig2.line(x, np.angle(y), line_width=2)
           else:
               fig1.renderers[0].data_source.data.update({'y': 20 * np.log10(abs(y))})
               fig2.renderers[0].data_source.data.update({'y': np.angle(y)})
           return pn.Column(
                   pn.pane.Bokeh(fig1, height=250),
                   pn.pane.Bokeh(fig2, height=250))
       else:
           if len(fig3.renderers) == 0:
               fig3.line(np.real(y), np.imag(y), line_width=2)
               fig3.line(np.real(y), -np.imag(y), line_width=2)
           else:
               fig3.renderers[0].data_source.data.update({'y': np.imag(y)})
               fig3.renderers[1].data_source.data.update({'y': -np.imag(y)})
           return pn.Column(pn.pane.Bokeh(fig3, height=500))

With ``@pn.depends(sliders[kp], sliders[t], sliders[z], plot_type)`` we are
explicitely asking for this function to be executed whenever we move the
sliders or click the buttons.
Note that ``update`` will receive the values of the specified widgets.

Next, we update the data series with the new parameters and extract the
numerical data from the symbolic transfer function:

.. code-block:: python

   s.update_data({
       kp: kpval,
       t: tval,
       z: zval,
   })
   x, y = s.get_data()

Note that we have passed in a dictionary of parameters, similarly to what we
would do if we were using ``iplot``.

The last thing to note is that the function returns the objects to be
updated: in our case, it will return the figures. The Bode plot is going to
use 2 figures, therefore the function returns 2 vertically aligned figures.
Nyquist plot will only use one figure.

Finally, we need to create the overall layout. Here, we'll use a
left-column containing the sliders and button, and a right-column containing
the plots:

.. code-block:: python

   pn.Row(
       pn.Column(sliders[kp], sliders[t], sliders[z], plot_type),
       update
   )


Example 2
=========

In this example we are going to explore a Non-Circular Planetary Drive.
The inspiration comes from the following resource, where we can also find
useful references:

    Erik Mahieu "Noncircular Planetary Drive"
    
    http://demonstrations.wolfram.com/NoncircularPlanetaryDrive/
    
    Wolfram Demonstrations Project
    
    Published: January 8 2014 


Let's see if we are able to create something similar with this module.

Let the angular motion of the driven gear be:

.. figure:: ../_static/tut-5/equation-2.png

where:

* ``t`` is the time;
* ``c`` is the number of lobes;
* ``r`` is the velocity ratio;
* ``p1, p2`` are displacement function parameters.

The time-derivative of the angular motion is a transfer function, that is,
the ratio of the driven and driving angular velocities:

.. figure:: ../_static/tut-5/equation-3.png

Let the angular velocity of the driving gear be ``omega1=1``. With a pure
rolling condition, the radii of the gears at the contact point are given by:

.. figure:: ../_static/tut-5/equation-4.png

Let's create an interactive model to study these relationships. We are going
to wrap everything into a function, ``NCPD``, which can be saved into a
Python file and later be called from any Jupyter notebook. The following code
is well documented, so let's explore it and then run it.

.. code-block:: python

   from spb.interactive import create_widgets
   import param

   def NCPD():
       # symbolic computations
       p1, p2, t, r, c = symbols("p1, p2, t, r, c")
       # driven angular motion
       phi = - (r * t + p1 * sin(c * r * t) + p2 * sin(2 * c * r * t))
       # transfer function
       phip = phi.diff(t)
       # profile of the driver gear
       r1 = phip / (1 + phip)
       # profile of the driven gear
       r2 = -r1 / phip
       
       # default parameter values
       params = { p1: 0.035, p2: 0.005, r:2, c:3 }
       
       # Create interactive data series: they will receive an updated
       # params dictionary each time a widget is modified.
       # Note: get_series=True returns a data series.
       #       get_series=False returns the numerical data.
       
       # line plot of the driven angular motion (function of a single
       # variable, t).
       s1 = get_plot_data(phi, (t, 0, 2 * pi),
                          get_series=True, pt="pinter", params=params)
       # line plot of the transfer function (function of a single
       # variable, t).
       s2 = get_plot_data(phip, (t, 0, 2 * pi),
                          get_series=True, pt="pinter", params=params)
       # line plot of the driver gear (function of a single variable, t).
       # Note that we are generating polar data.
       s3 = get_plot_data(r1, (t, 0, 2 * pi), polar=True,
                          get_series=True, pt="pinter", params=params)
       # line plot of the driven gear (function of a single variable, t).
       # Here, the function is polar wrt to the driven angular motion, Phi,
       # which is symbolically computed.
       s4 = get_plot_data(r2 * cos(phi), r2 * sin(phi), (t, 0, 2 * pi),
                          get_series=True, pt="pinter", params=params)
       # Numerical rotation matrix about the z-axis.
       Rz = lambda k: np.array([[np.cos(k), np.sin(k)], [-np.sin(k), np.cos(k)]])

       
       # manually create widgets: this allows for full customization
       from bokeh.models.formatters import PrintfTickFormatter
       # by default, float sliders only shows numbers up to the second
       # decimal place. Use PrintfTickFormatter to customize the printed value.
       p1s = pn.widgets.FloatSlider(name="p1", start=-0.035, end=0.035,
                value=params[p1],
                step=0.001, format=PrintfTickFormatter(format='%.3f'))
       p2s = pn.widgets.FloatSlider(name="p2", start=-0.02, end=0.02,
                value=params[p2],
                step=0.001, format=PrintfTickFormatter(format='%.3f'))
       ts = pn.widgets.FloatSlider(name="Time, t", start=0, end=2 * np.pi,
                value=0, step=0.05)
       rs = pn.widgets.IntSlider(name="Speed ratio, r", start=2, end=5,
                value=params[r])
       c_btns = pn.widgets.RadioButtonGroup(
           name='Number of lobes, c',
           options=[1, 2, 3, 4, 5],
           value=params[c],
           button_type='primary')
       
       # layout the control widgets
       widgets_col = pn.Column(
           ts,
           rs,
           # NOTE: some panel's widgets have a bug: they do not display the
           # name. Hence, we need to add a custom label to let the user
           # know what the widget is representing.
           pn.pane.HTML("<div>Number of lobes, c</div>"),
           c_btns,
           p1s,
           p2s
       )
       
       # create two empty figures
       fig1 = plot(backend=BB, xlabel="t", ylabel="Phi",
                   title="Driven Angular Motion + Transfer Function",
                   show=False).fig
       fig2 = plot(backend=BB, xlabel="x", ylabel="y",
                   title="Rolling Curves",
                   aspect="equal", show=False).fig
       
       # customize the first figure: two y-axis, the left one with
       # blue color, the right one with red color
       from bokeh.models import Range1d, LinearAxis
       color1, color2 = "blue", "red"
       fig1.yaxis.axis_line_color = color1
       fig1.yaxis.major_label_text_color = color1
       fig1.yaxis.major_tick_line_color = color1
       fig1.yaxis.minor_tick_line_color = color1
       fig1.yaxis.axis_label_text_color = color1
       
       fig1.extra_y_ranges = {'phip': Range1d(start=0, end=1)}
       fig1.add_layout(
           LinearAxis(
               y_range_name='phip',
               axis_label='d(Phi)/dt',
               axis_line_color = color2,
               major_label_text_color = color2,
               major_tick_line_color = color2,
               minor_tick_line_color = color2,
               axis_label_text_color = color2
           ), 
           'right')
       
       # update the data series with new parameters and generate
       # new data. this function reduces code repetition.
       def get_data(series, d):
           series.update_data(d)
           return series.get_data()
       
       # callback function.
       @pn.depends(ts, rs, c_btns, p1s, p2s)
       def update(tval, rval, cval, p1val, p2val):
           # substitution dictionary
           d = {p1: p1val, p2: p2val, r: rval, c: cval}
           # updated values
           x1, y1 = get_data(s1, d)
           source1 = {"xs": x1, "ys": y1}
           x2, y2 = get_data(s2, d)
           source2 = {"xs": x2, "ys": y2}
           x3, y3 = get_data(s3, d)
           source3 = {"xs": x3, "ys": y3}
           x4, y4, _ = get_data(s4, d)
           
           # apply the rotations to the driven gear
           d[t] = tval
           # NOTE: the following two evaluations are slow because they are
           # computed with SymPy. Alternatively, we could create two lambda
           # functions outside of this update function to speed things up.
           angle = float(phi.evalf(subs=d))
           phipval = float(phip.evalf(subs=d))
           x4, y4 = np.matmul(Rz(angle), np.array([x4, y4]))
           x4 += 1
           x4, y4 = np.matmul(Rz(tval), np.array([x4, y4]))
           source4 = {"xs": x4, "ys": y4}
           # points on fig1 to visualize the current location
           source5 = {"xs": [tval], "ys": [angle]}
           source6 = {"xs": [tval], "ys": [phipval]}

           if len(fig1.renderers) == 0:
               # add data to the figures
               fig1.line("xs", "ys", source=source1, color=color1)
               fig1.line("xs", "ys", source=source2, color=color2,
                    y_range_name="phip")
               fig2.line("xs", "ys", source=source3, color=color1)
               fig2.line("xs", "ys", source=source4, color=color2)
               # add two dots representing the current time
               fig1.circle("xs", "ys", source=source5, color="black")
               fig1.circle("xs", "ys", source=source6, color="black",
                    y_range_name="phip")
           else:
               # update data
               fig1.renderers[0].data_source.data.update(source1)
               fig1.renderers[1].data_source.data.update(source2)
               fig1.renderers[2].data_source.data.update(source5)
               fig1.renderers[3].data_source.data.update(source6)
               fig2.renderers[0].data_source.data.update(source3)
               fig2.renderers[1].data_source.data.update(source4)

           # vertocally center the data of fig1, leaving 5% of whitespace on
           # top and bottom
           m1, M1 = min(y1), max(y1)
           offset1 = abs(M1 - m1) * 0.05
           m2, M2 = min(y2), max(y2)
           offset2 = abs(M2 - m2) * 0.05
           fig1.y_range.update(start=m1-offset1, end=M1+offset1)
           fig1.extra_y_ranges['phip'].update(start=m2-offset2, end=M2+offset2)

           # layout for the figures
           return pn.Column(
               pn.pane.Bokeh(fig1, height=200), 
               pn.pane.Bokeh(fig2),
               width=500
           )
       
       # return the overall layout
       return pn.Row(
           widgets_col,
           update
       )

Note that we used the ``create_widgets`` function to quickly create sliders
and save us from some typing, as it understands the same syntax passed to
``iplot`` parameters.

Let's play with the model:

.. code-block:: python

   NCPD()

As we can see, as soon as we move any slider, an update will be executed.
What if our computation takes a long time? In these occasions, it might be
better to execute an update only when the click is released from the slider:
this will improve user experience as there won't be significant lags.
To achieve this behaviour, we have to change the following line of code:

.. code-block:: python

   @pn.depends(ts, rs, c_btns, p1s, p2s)

to something like the following:

.. code-block:: python

   @pn.depends(ts.param.value_throttled, rs, c_btns, p1s, p2s)

Obviously, we should attach ``.param.value_throttled`` to all sliders.
This is left to the Reader as an exercise.


Finally, a couple of observations:

1. If we are building complicated applications with several widgets and plots,
   we can again capture the output ``panel`` object and launch a server by
   calling the ``show()`` method. For example, ``NCPD().show()`` will open
   the interactive application on a new browser window. In doing so, we can
   make better use of all the available space.
2. The creation of these interactive widgets is a trial and error procedure.
   We will have to execute the same code blocks over and over again, generating
   new figures, new widgets, etc. Memory consumption constantly goes up, so it
   is a good idea to keep an eye on our system resource monitor. If the
   browser starts lagging, or memory consumption is too high, try to close
   the browser, close Jupyter server and starts over.
