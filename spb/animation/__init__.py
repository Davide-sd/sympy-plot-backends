from imageio.v3 import imwrite, imread
from imageio import mimwrite
import os
from spb.animation.animation import AnimationData, BaseAnimation
from tempfile import TemporaryDirectory
from tqdm.notebook import trange


class SaveAnimation:
    def save(self, path, **kwargs):
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
    
        ext = os.path.splitext(path)[1]

        with TemporaryDirectory(prefix="animation") as tmpdir:
            filenames = []
            n_frames = self._backend._animation_data.n_frames
            fps = self._backend._animation_data.fps

            for i in trange(n_frames):
                self._backend.update_animation(i)
                filename = os.path.join(tmpdir, str(i) + ".png")
                filenames.append(filename)
                self._backend.save(filename)

            frames = [imread(f) for f in filenames]

            if ext == ".gif":
                kwargs.setdefault("duration", int(1000 / fps))
                imwrite(path, frames, **kwargs)
            elif ext == ".mp4":
                kwargs.setdefault("fps", fps)
                kwargs.setdefault("quality", 10)
                mimwrite(path, frames, **kwargs)
            else:
                mimwrite(path, frames, **kwargs)
