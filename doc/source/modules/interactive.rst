.. _interactive:

Interactive
-----------

This module allows the creation of interactive widgets plots using either one
of the following modules:

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
with ``ipywidgets``. To use Holoviz's Panel, users can either: 

1. specify the interactive module at runtime by providing the following
   keyword argument: ``imodule="panel"``.
2. Modify the configuration file to permanently change the interactive module.
   More information are provided in :ref:`Tutorial 4`.

Note that interactive capabilities are already integrated with many
plotting functions. The purpose of the following documentation is to show a
few more examples for each interactive module.


Holoviz's panel
===============

.. module:: spb.interactive.panel

.. autofunction:: iplot


ipywidgets
==========

.. module:: spb.interactive.ipywidgets

.. autofunction:: iplot