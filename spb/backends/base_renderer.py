from spb.backends.base_backend import Plot


class Renderer:
    """Base class for renderers. A renderer is responsible to render something
    on a plot and keep it updated when interactive widgets change states.

    A renderer stores the following information:

    * ``plot``: the instance of the ``Plot`` class containing the actual
      figure in which the renderer adds graphical elements.
    * ``series``: the instance of BaseSeries that generates numerical data.
    * ``handles``: a list, in which each handle stores the necessary objects
      in order to update the graphical elements in case of interactive-widgets
      plot. It will be populated by the ``renderer.draw()`` method.
    * ``draw_update_map``: a dictionary mapping `draw_method` to
      `update_method`, which is where the rendering is actually implemented.
      In particular:

      * ``draw_method(renderer, data)``: use numerical ``data`` to add
        graphical elements to ``renderer.plot.fig``. It must return an handle,
        which is eventually used by `update_method`.
      * ``update_method(renderer, data, handle)``: update graphical elements
        stored in ``handle`` with new numerical ``data``.

      Multiple key/value pairs can be added, all of which will receive the
      same numerical data. This allows to add different graphical elements
      to the same data series, in order to create more complex plots, promoting
      code reusability at the same time.

    A renderer implements these methods:

    * `draw`: it will be called by ``plot`` when the figure is empty. This
      method extracts the numerical data from the ``series``, and sends it to
      each ``draw_method`` contained in ``draw_update_map``.
    * ``update``: it will be called by ``plot`` when the widgets change state.
      This method extracts the numerical data from the ``series``, and sends it
      to each ``update_method`` contained in ``draw_update_map``, together
      with the appropriate ``handle``.
    """
    draw_update_map = {}

    def __init__(self, plot, series):
        self.plot = plot if isinstance(plot, Plot) else plot.backend
        self.series = series
        self.handles = []

    def draw(self):
        data = self.series.get_data()
        for draw_method in self.draw_update_map.keys():
            self.handles.append(
                draw_method(self, data))

    def update(self, params):
        if self.series.is_interactive:
          self.series.params = params
        data = self.series.get_data()
        for update_method, handle in zip(
          self.draw_update_map.values(), self.handles):
            update_method(self, data, handle)
