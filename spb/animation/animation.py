from sympy import Symbol
from sympy.external import import_module


# import moviepy
# https://community.plotly.com/t/how-to-export-animation-and-save-it-in-a-video-format-like-mp4-mpeg-or/64621/2


class BaseAnimation:
    """This base class is meant to be used by ``Plot`` and ``PlotGrid``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._animation_data = None
    
    def update_animation(self, frame_idx):
        """Update the figure in order to obtain the visualization at a
        specifie frame of the animation.
        """
        if not self._animation_data:
            raise RuntimeError(
                "The data necessary to build the animation has not been "
                "provided. You must set `animation=True` to the function call."
            )
        params = self._animation_data[frame_idx]
        self.update_interactive(params)


class AnimationData:
    """Verify that the user provided the appropriate parameters. If so,
    creates a matrix with the following form:

    .. code-block:: text

               | param 1 | param 2 | ... | param M |
       --------|------------------------------------
       frame 1 |   val   |   val   | ... |   val   |
       frame 2 |   val   |   val   | ... |   val   |
        ...    |   ...   |   ...   | ... |   ...   |
       frame N |   val   |   val   | ... |   val   |

    Where each column represents a time-series of values associated to a
    particular symbol. Each row represent the values of all symbols at
    a particular time.
    """
    def __init__(self, fps=30, time=5, params=None):
        if not isinstance(params, dict):
            raise TypeError("``params`` must be a dictionary.")
        if len(params) == 0:
            raise ValueError(
                "In order to build an animation, at lest one "
                "parameter must be provided.")
        not_symbols = [not isinstance(k, Symbol) for k in params.keys()]
        if any(not_symbols):
            raise ValueError(
                "All keys of ``params`` must be a single symbol. The "
                "following keys are something else: %s" % [k for i, k in
                    enumerate(params.keys()) if not_symbols[i]])
        
        self.parameters = list(params.keys())
        self.n_frames = fps * time
        self.time = time
        self.fps = fps

        np = import_module("numpy")
        values = []
        for k, v in params.items():
            if isinstance(v, dict):
                values.append(self._create_steps(v))
            elif isinstance(v, (list, tuple)) and (len(v) <= 3):
                values.append(self._create_interpolation(v))
            elif isinstance(v, np.ndarray):
                if len(v) != self.n_frames:
                    raise ValueError(
                        "The length of the values associated to `%s` must "
                        "be %s. Instead, an array of length %s was received."
                        "" % (k, self.n_frames, len(v))
                    )
                values.append(v)
            else:
                raise TypeError(
                    "The value associated to '%s' is not supported. "
                    "Expected an instance of `dict`, or `list` or `tuple`. "
                    "Received: %s" % (k, type(v))
                )
        
        self.matrix = np.array(values).T
    
    def _create_steps(self, d):
        np = import_module("numpy")
        ani_time = self.time
        values = np.zeros(self.n_frames)
        for time, val in d.items():
            n_frame = int(time / ani_time * self.n_frames)
            values[n_frame:] = val
        return values

    def _create_interpolation(self, v):
        if len(v) == 2:
            start, end = v
            strategy = "linear"
        else:
            start, end, strategy = v
            strategy = strategy.lower()

        allowed_strategies = ["linear", "log"]
        if not (strategy in allowed_strategies):
            raise ValueError(
                "Discretization strategy must be either one of the "
                "following: %s" % allowed_strategies)
        
        np = import_module("numpy")
        if strategy == "linear":
            return np.linspace(start, end, self.n_frames)
        return np.geomspace(start, end, self.n_frames)
    
    def __getitem__(self, index):
        """Returns a dictionary mapping parameters to values at the specified
        animation frame.
        """
        return {k: v for k, v in zip(self.parameters, self.matrix[index, :])}
        
