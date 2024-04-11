.. _interactive:

Interactive
-----------

The aim of this module is to quickly and easily create interactive widgets
plots using either one of the following modules:

* ``ipywidgets`` and ``ipympl``: it probably has less dependencies to install
  and it might be a little bit faster at updating the visualization. However,
  it only works inside Jupyter Notebook.

* Holoviz's ``panel``: works on any Python interpreter, as
  long as a browser is installed on the system. The interactive plot can be
  visualized directly in Jupyter Notebook, or in a new browser window where
  the plot can use the entire screen space. This might be useful to visualize
  plots with many parameters, or to focus the attention only on the plot rather
  than the code.

If only a minimal installation of this plotting module has been performed,
then users have to manually install the chosen interactive modules.

By default, this plotting module will attempt to create interactive widgets
with ``ipywidgets``. To change interactive module, users can either: 

1. specify the following keyword argument to use Holoviz's Panel:
   ``imodule="panel"``. By default, the modules uses ``imodule="ipywidgets"``.
2. Modify the configuration file to permanently change the interactive module.
   More information are provided in :ref:`Tutorial 4`.

To create an interactive widget plot, users have to provide the ``params=``
keyword argument to a plotting function, which must be a dictionary mapping
parameters (SymPy's symbols) to widgets. Let ``a, b, c, d`` be parameters.
When ``imodule="panel"``, an example of params-dictionary is:

.. code-block:: python

   import panel as pn
   params = {
       a: (1, 0, 5), # slider from 0 to 5, with default value of 1
       b: pn.widgets.FloatSlider(value=1, start=0, end=5), # same slider as above
       (c, d): pn.widgets.RangeSlider(value=(-1, 1), start=-3, end=3, step=0.1)
   }

Or with ``imodule="ipywidgets"``:

.. code-block:: python

   import ipywidgets as w
   params = {
       a: (1, 0, 5), # slider from 0 to 5, with default value of 1
       b: w.FloatSlider(value=1, min=0, max=5), # same slider as above
       (c, d): w.FloatRangeSlider(value=(-1, 1), min=-3, max=3, step=0.1)
   }

From the above examples, it can be seen that the module requires widgets that
returns numerical values. The code ``a: (1, 0, 5)`` is a shortcut to create
a slider (more information down below). If a widget returns multiple
numerical values (like
:py:class:`panel.widgets.slider.RangeSlider` or
:py:class:`ipywidgets.widgets.widget_float.FloatRangeSlider`),
then a corresponding number of symbols must be provided.

Note that:

* Interactive capabilities are already integrated with many plotting functions.
  The purpose of the following documentation is to show a few more examples
  for each interactive module.

* It is not possible to mix ipywidgets and panel.

* If the user is attempting to execute an interactive widget plot and gets an
  error similar to the following:
  *TraitError: The 'children' trait of a Box instance contains an Instance of
  a TypedTuple which expected a Widget, not the FigureCanvasAgg at '0x...'*.
  It means that the ipywidget module is being used with Matplotlib, but the
  interactive Matplotlib backend has not been loaded. First, execute the magic
  command ``%matplotlib widget``, then execute the plot command.

* For technical reasons, all interactive-widgets plots in this documentation
  are created using Holoviz's Panel. Often, they will ran just fine with
  ipywidgets too. However, if a specific example uses the ``param`` library,
  or widgets from the ``panel`` module, then users will have to modify the
  ``params`` dictionary in order to make it work with ipywidgets.
  A couple of examples are provided below.


The prange class
================

.. module:: spb.utils

.. autoclass:: prange


Holoviz's panel
===============

.. module:: spb.interactive.panel

.. autofunction:: iplot


ipywidgets
==========

.. module:: spb.interactive.ipywidgets

.. autofunction:: iplot
