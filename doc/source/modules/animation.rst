Animations
----------

The aim of this module is to quickly and easily create animations. Here,
an animation is a kind of interactive-widgets plot, which contains a figure
and widgets to control the playback. Animations can be created using either
one of the following modules:

* ``ipywidgets`` and ``ipympl``: it only works inside Jupyter Notebook.

* Holoviz's ``panel``: works on any Python interpreter, as
  long as a browser is installed on the system. The interactive plot can be
  visualized directly in Jupyter Notebook, or in a new browser window where
  the plot can use the entire screen space.

If only a minimal installation of this plotting module has been performed,
then users have to manually install the chosen interactive modules.

By default, this plotting module will attempt to create animations
with ``ipywidgets``. To change interactive module, users can either: 

1. specify the following keyword argument to use Holoviz's Panel:
   ``imodule="panel"``. By default, the modules uses ``imodule="ipywidgets"``.
2. Modify the configuration file to permanently change the interactive module.
   More information are provided in :ref:`Tutorial 4`.

Note that:

* To create an animation, users have to provide the ``params=`` and
  ``animation=`` keyword arguments to a plotting function. In particular,
  ``params=`` is different from what is used in the :ref:`interactive` module.
  More on this in the documentation below.

* If the user is attempting to execute an interactive widget plot and gets an
  error similar to the following:
  *TraitError: The 'children' trait of a Box instance contains an Instance of
  a TypedTuple which expected a Widget, not the FigureCanvasAgg at '0x...'*.
  It means that the ipywidget module is being used with Matplotlib, but the
  interactive Matplotlib backend has not been loaded. First, execute the magic
  command ``%matplotlib widget``, then execute the plot command.

* Tha aim of this module is to remain as simple as possible, while using the
  already implemented code that makes interactive-widgets plots possible.
  If a user needs more customization options, then
  `Manim <https://github.com/ManimCommunity/manim/>`_ is a far more
  powerful alternative.


.. module:: spb.animation.panel

.. autofunction:: animation

.. module:: spb.animation

.. autoclass:: spb.animation.BaseAnimation

.. autofunction:: spb.animation.BaseAnimation.get_FuncAnimation

.. autofunction:: spb.animation.BaseAnimation.save

.. autofunction:: spb.animation.BaseAnimation.update_animation
