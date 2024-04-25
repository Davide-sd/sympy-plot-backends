from imageio.v3 import imwrite, imread
from imageio import mimwrite
import io
import os
import shutil
from spb.interactive import IPlot
from spb.utils import _aggregate_parameters, get_environment
from sympy import Symbol
from sympy.external import import_module
from tempfile import TemporaryDirectory
from tqdm.notebook import trange


class BaseAnimation:
    """Implements the base functionalities to create animations.
    """

    def _post_init_plot(self, *args, **kwargs):
        """This methods has to be executed after self._backend has been set.
        """
        mergedeep = import_module("mergedeep")
        merge = mergedeep.merge
        animation = kwargs.get("animation", False)

        self._animation_data = None
        if isinstance(animation, AnimationData):
            self._animation_data = animation
        else:
            params = {}
            if animation:
                params = _aggregate_parameters(params, self._backend.series)
            animation_data_kwargs = {"params": params}
            if isinstance(animation, dict):
                animation_data_kwargs = merge({},
                    animation_data_kwargs, animation)
            if animation:
                self._animation_data = AnimationData(**animation_data_kwargs)
                initial_params = self.animation_data[0]
                # update series with proper initial values before plotting
                for s in self._backend.series:
                    if s.is_interactive:
                        s.params = initial_params

    def _post_init_plotgrid(self, *args, **kwargs):
        """This methods has to be executed after self._backend has been set.
        """
        self._animation_data = None
        original_params, fps, time = {}, [], []
        for p in self._backend._all_plots:
            if isinstance(p, BaseAnimation):
                original_params = _aggregate_parameters(
                    original_params.copy(), p.backend.series)
                fps.append(p._animation_data.fps)
                time.append(p._animation_data.time)
        if original_params:
            self._animation_data = AnimationData(
                fps=max(fps), time=max(time), params=original_params)

    @property
    def animation_data(self):
        return self._animation_data

    def update_animation(self, frame_idx):
        """Update the figure in order to obtain the visualization at a
        specified frame of the animation.

        Parameters
        ==========
        frame_idx : int
            Must be ``0 <= frame_idx < fps*time``.
        """
        if not self.animation_data:
            raise RuntimeError(
                "The data necessary to build the animation has not been "
                "provided. You must set `animation=True` to the function call."
            )
        params = self.animation_data[frame_idx]
        self.backend.update_interactive(params)

    def get_FuncAnimation(self):
        """Return a Matplotlib's ``FuncAnimation`` object. It only works if
        this animation is showing a Matplotlib's figure.
        """
        from spb import MB
        if not isinstance(self.backend, MB):
            raise TypeError(
                "FuncAnimation can only be created when the backend produced "
                "Matplotlib's figure. "
                f"`{type(self.backend).__name__}` does not."
            )
        matplotlib = import_module(
            'matplotlib',
            import_kwargs={
                'fromlist': ['animation']
            },
            warn_not_installed=True,
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        return matplotlib.animation.FuncAnimation(
            fig=self.backend.fig, func=self.update_animation,
            frames=self.animation_data.n_frames,
            interval=int(1000 / self.animation_data.fps)
        )

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

            * GIFs are created with
              :py:class:`imageio.plugins.pillowmulti.GIFFormat`
            * MP4s are created with
              :py:class:`imageio.plugins.ffmpeg.FfmpegFormat`. If a video seems
              to be low-quality, try to increase the bitrate. Its default
              value is ``bitrate=3000000``.

        Notes
        =====

        Saving K3D-Jupyter animations is particularly slow.

        """
        ext = os.path.splitext(path)[1]
        if len(ext) == 0:
            raise ValueError("Please, provide a file extension.")

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
        n_frames = self.animation_data.n_frames
        base = os.path.basename(path).split(".")[0]

        @self._backend.fig.yield_screenshots
        def inner_func():
            frames = []
            r = (range(n_frames) if get_environment() != 0
                else trange(n_frames))
            for i in r:
                self.update_animation(i)
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
        n_frames = self.animation_data.n_frames
        base = os.path.basename(path).split(".")[0]

        with TemporaryDirectory(prefix="animation") as tmpdir:
            tmp_filenames = []
            dest = os.path.dirname(path)
            if dest == "":
                dest = "."

            r = (range(n_frames) if get_environment() != 0
                else trange(n_frames))
            for i in r:
                self.update_animation(i)
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
        fps = self.animation_data.fps
        if ext == ".gif":
            kwargs.setdefault("loop", 0)    # loop=0 means loops continuously
            kwargs.setdefault("fps", fps)
            # NOTE: from my tests on 3D plots with colorbars, 2 works best.
            kwargs.setdefault("quantizer", 2)
        elif ext == ".mp4":
            kwargs.setdefault("fps", fps)
            # NOTE: setting quality=something would use variable bitrate.
            # However, this creates artifacts between consecutive frames.
            # Instead, let's use constant bitrate.
            kwargs.setdefault("bitrate", 3000000)
        mimwrite(path, frames, **kwargs)


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
        self.n_frames = int(fps * time)
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
