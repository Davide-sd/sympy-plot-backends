from imageio.v3 import imwrite, imread
from imageio import mimwrite
import io
import os
import shutil
from sympy import Symbol
from sympy.external import import_module
from tempfile import TemporaryDirectory
from tqdm.notebook import trange


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
            start, end = [float(t) for t in v]
            strategy = "linear"
        else:
            start, end, strategy = v
            start, end = float(start), float(end)
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


class SaveAnimation:
    """Implement the logic to save animations.
    """

    def save(self, path, save_frames=False, **kwargs):
        """Save the animation to a file.

        Parameters
        ==========

        path : str
            Where to save the animation on the disk. Supported formats are
            ``.gif`` or ``.mp4``.
        save_frames : bool, optional
            Default to False. If True, save individual frames into png files.
        **kwargs :
            Keyword arguments to customize the gif/video creation process.
            Both gif/video animations are created using ``imageio.mimwrite``.
            In particular:

            * gif files are created with
              :py:class:`imageio.plugins.pillowmulti.GIFFormat`
            * gif files are created with
              :py:class:`imageio.plugins.ffmpeg.FfmpegFormat`. If a video seems
              to be low-quality, try to increase the bitrate. Its default
              value is ``bitrate=3000000``.

        Notes
        =====

        Saving K3D-Jupyter animations is particularly slow.

        """
        # avoid circular imports
        from spb.plotgrid import PlotGrid
        if (
            isinstance(self._backend, PlotGrid) and
            (not self._backend.is_matplotlib_fig)
        ):
            raise RuntimeError(
                "Saving plotgrid animation is only supported when the overall "
                "figure is a Matplotlib's figure."
            )

        from spb import KB
        if isinstance(self._backend, KB):
            self._save_k3d_animation(path, save_frames, **kwargs)
        else:
            self._save_other_backends_animation(path, save_frames, **kwargs)

    def _save_k3d_animation(self, path, save_frames, **kwargs):
        n_frames = self._backend._animation_data.n_frames
        base = os.path.basename(path).split(".")[0]

        @self._backend.fig.yield_screenshots
        def inner_func():
            frames = []
            for i in trange(n_frames):
                self._backend.update_animation(i)
                self._backend.fig.fetch_screenshot()
                screenshot_bytes = yield
                buffer = io.BytesIO(screenshot_bytes)
                img = imread(buffer)
                frames.append(img)
                if save_frames:
                    name = base + "_" + str(i) + ".png"
                    imwrite(os.path.join(os.path.dirname(path), name), img)
            self._save_helper(path, frames, **kwargs)

        inner_func()

    def _save_other_backends_animation(self, path, save_frames, **kwargs):
        n_frames = self._backend._animation_data.n_frames
        base = os.path.basename(path).split(".")[0]

        with TemporaryDirectory(prefix="animation") as tmpdir:
            tmp_filenames = []
            dest = os.path.dirname(path)
            if dest == "":
                dest = "."

            for i in trange(n_frames):
                self._backend.update_animation(i)
                filename = base + "_" + str(i) + ".png"
                tmp_filename = os.path.join(tmpdir, filename)
                tmp_filenames.append(tmp_filename)
                self._backend.save(tmp_filename)
                if save_frames:
                    shutil.copy2(tmp_filename, dest)

            frames = [imread(f) for f in tmp_filenames]

        self._save_helper(path, frames, **kwargs)


    def _save_helper(self, path, frames, **kwargs):
        ext = os.path.splitext(path)[1]
        fps = self._backend._animation_data.fps
        if ext == ".gif":
            kwargs.setdefault("loop", True)
            kwargs.setdefault("fps", fps)
        elif ext == ".mp4":
            kwargs.setdefault("fps", fps)
            # NOTE: setting quality=something would use variable bitrate.
            # However, this creates artifacts between consecutive frames.
            # Instead, let's use constant bitrate.
            kwargs.setdefault("bitrate", 3000000)
        mimwrite(path, frames, **kwargs)
