from spb.backends.base_renderer import Renderer


class MatplotlibRenderer(Renderer):
    """A base class for renderers related to Matplotlib.

    Matplotlib is not really great at keeping track of axis limits when
    Collections (LineCollection, PolyCollection, ...). This base class
    implements the code to compute them.
    """
    draw_update_map = {}

    # NOTE: matplotlib 3d plots (and also 2D plots containing LineCollection)
    # suffer from this problem:
    # https://github.com/matplotlib/matplotlib/issues/17130
    # Renderers that deals with Matplotlib's `*Collection` objects should
    # set this attribute to True, which is going to compute axis limits
    # from the numerical data.
    _sal = False

    def __init__(self, plot, s):
        super().__init__(plot, s)
        self._xlims = []
        self._ylims = []
        self._zlims = []

    def draw(self):
        data = self.series.get_data()
        self._set_axis_limits(data)
        for draw_method in self.draw_update_map.keys():
            self.handles.append(
                draw_method(self, data))

    def update(self, params):
        self.series.params = params
        data = self.series.get_data()
        self._set_axis_limits(data)
        for update_method, handle in zip(
            self.draw_update_map.values(), self.handles
        ):
            update_method(self, data, handle)

    def _set_axis_limits(self, data):
        np = self.plot.np
        if self._sal:
            # NOTE: matplotlib's axis limits cannot be NaN or Inf
            self._xlims = [[np.nanmin(data[0]), np.nanmax(data[0])]]
            self._ylims = [[np.nanmin(data[1]), np.nanmax(data[1])]]
            if self.series.is_3D:
                self._zlims = [[np.nanmin(data[2]), np.nanmax(data[2])]]
