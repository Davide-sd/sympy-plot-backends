from matplotlib.animation import FuncAnimation
import os
import pytest
from spb import plot, line, graphics, MB, PB, BB, KB, surface
from spb.animation.ipywidgets import Animation as IPYAnimation
from spb.animation.panel import Animation as PanelAnimation
from sympy import cos, sin, exp, pi
from sympy.abc import a, b, x, y
from tempfile import TemporaryDirectory


@pytest.mark.parametrize(
    "backend, imodule, expected_type", [
        (MB, "panel", PanelAnimation),
        (MB, "ipywidgets", IPYAnimation),
        (PB, "panel", PanelAnimation),
        (PB, "ipywidgets", IPYAnimation),
        (BB, "panel", PanelAnimation),
        (BB, "ipywidgets", IPYAnimation),
    ]
)
def test_animation_single_plot_1d(backend, imodule, expected_type):
    # default animation length/fps
    p = graphics(
        line(cos(a*x), params={a: (1, 2)}, n=10),
        animation=True, show=False,
        imodule=imodule, backend=backend
    )
    assert isinstance(p, expected_type)
    animation_data = p._animation_data
    fps = 30
    time = 5
    assert animation_data.matrix.shape == (fps * time, 1)

    # custom animation length/fps
    p = graphics(
        line(b*cos(a*x), params={a: (1, 2), b: (3, 4)}, n=10),
        animation={"fps": 5, "time": 2},
        show=False, imodule=imodule
    )
    animation_data = p._animation_data
    fps = 5
    time = 2
    assert animation_data.matrix.shape == (fps * time, 2)
    # run all frames by saving the animation. no error should be raised.
    with TemporaryDirectory(prefix="animation_1d") as tmpdir:
        p.save(os.path.join(tmpdir, "animation_1.gif"))
        assert len(os.listdir(tmpdir)) == 1

        # number of files: 1 animation from previous command + 1 animation
        # and frames from this command.
        p.save(os.path.join(tmpdir, "animation_2.mp4"), save_frames=True)
        assert len(os.listdir(tmpdir)) == 1 + fps * time + 1


@pytest.mark.parametrize(
    "backend, imodule", [
        (MB, "panel"),
        (MB, "ipywidgets"),
        (PB, "panel"),
        (PB, "ipywidgets"),
        (BB, "panel"),
        (BB, "ipywidgets"),
    ]
)
def test_animation_single_plot_1d_get_func_animation(backend, imodule):
    p = graphics(
        line(cos(a*x), params={a: (1, 2)}, n=10),
        animation=True, show=False,
        imodule=imodule, backend=backend
    )
    if backend is MB:
        assert isinstance(p.get_FuncAnimation(), FuncAnimation)
    else:
        with pytest.raises(
            TypeError,
            match="FuncAnimation can only be created when the backend produced"
        ):
            p.get_FuncAnimation()


# NOTE: slow test, especially Plotly
@pytest.mark.parametrize(
    "backend, imodule, expected_type", [
        (MB, "panel", PanelAnimation),
        (MB, "ipywidgets", IPYAnimation),
        (PB, "panel", PanelAnimation),
        (PB, "ipywidgets", IPYAnimation),
        (KB, "panel", PanelAnimation),
        (KB, "ipywidgets", IPYAnimation),
    ]
)
def test_animation_single_plot_3d(backend, imodule, expected_type):
    p = graphics(
        surface(
            cos(x**2 + y**2 - a) * exp(-(x**2 + y**2) * 0.2),
            (x, -pi, pi), (y, -pi, pi),
            params={a: (1, 2)}, n=10
        ),
        animation={"fps": 3, "time": 1}, # NOTE: keep them down for performance
        show=False, imodule=imodule, backend=backend
    )
    assert isinstance(p, expected_type)
    animation_data = p._animation_data
    fps = 3
    time = 1
    assert animation_data.matrix.shape == (fps * time, 1)
    if backend is not KB:
        # run all frames by saving the animation. no error should be raised.
        with TemporaryDirectory(prefix="animation_3d") as tmpdir:
            p.save(os.path.join(tmpdir, "animation_1.gif"))
            assert len(os.listdir(tmpdir)) == 1

            # number of files: 1 animation from previous command + 1 animation
            # and frames from this command.
            p.save(os.path.join(tmpdir, "animation_2.mp4"), save_frames=True)
            assert len(os.listdir(tmpdir)) == 1 + fps * time + 1


@pytest.mark.parametrize(
    "backend, imodule", [
        (MB, "panel"),
        (MB, "ipywidgets"),
        (PB, "panel"),
        (PB, "ipywidgets"),
        (KB, "panel"),
        (KB, "ipywidgets"),
    ]
)
def test_animation_single_plot_3d_get_func_animation(backend, imodule):
    p = graphics(
        surface(
            cos(x**2 + y**2 - a) * exp(-(x**2 + y**2) * 0.2),
            (x, -pi, pi), (y, -pi, pi),
            params={a: (1, 2)}, n=10
        ),
        animation={"fps": 3, "time": 1}, # NOTE: keep them down for performance
        show=False, imodule=imodule, backend=backend
    )
    if backend is MB:
        assert isinstance(p.get_FuncAnimation(), FuncAnimation)
    else:
        with pytest.raises(
            TypeError,
            match="FuncAnimation can only be created when the backend produced"
        ):
            p.get_FuncAnimation()
