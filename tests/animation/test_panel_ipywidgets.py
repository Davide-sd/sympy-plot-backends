from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import os
import pytest
ipywidgets = pytest.importorskip("ipywidgets")
from spb import (
    plot, line, graphics, MB, PB, BB, KB, surface, plotgrid, plot3d,
    prange
)
from spb.animation.ipywidgets import Animation as IPYAnimation
from spb.animation.panel import Animation as PanelAnimation
from spb.plotgrid import PlotGrid
from sympy import cos, sin, exp, pi
from sympy.abc import a, b, c, x, y
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
    animation_data = p.animation_data
    fps = 30
    time = 5
    assert animation_data.matrix.shape == (fps * time, 1)

    # custom animation length/fps
    p = graphics(
        line(b*cos(a*x), params={a: (1, 2), b: (3, 4)}, n=10),
        animation={"fps": 5, "time": 2},
        show=False, imodule=imodule
    )
    animation_data = p.animation_data
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
    animation_data = p.animation_data
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


@pytest.mark.parametrize(
    "imodule, expected_type", [
        ("ipywidgets", IPYAnimation),
        ("panel", PanelAnimation),
    ]
)
def test_plotgrid_mode_1_matplotlib_animation(imodule, expected_type):
    options = dict(show=False, n=10, backend=MB, imodule=imodule)
    p1 = plot(cos(a*x), (x, -3*pi, 3*pi),
        params={a: (1, 5)}, animation={"fps": 10, "time": 1}, **options)
    p2 = plot(sin(b*x), (x, -3*pi, 3*pi),
        params={b: (2, 4)}, animation={"fps": 4, "time": 2}, **options)
    p = plotgrid(p1, p2, show=False)
    assert isinstance(p, expected_type)
    assert isinstance(p.backend, PlotGrid)
    # verify that plotgrid collects all parameters from its plots,
    # and that the max fps/time are used
    animation_data = p.animation_data
    fps = 10
    time = 2
    assert animation_data.matrix.shape == (fps * time, 2)
    # no error should be raised when saving the animation
    with TemporaryDirectory(prefix="animation_plogrid_1") as tmpdir:
        p.save(os.path.join(tmpdir, "animation.gif"))
        assert len(os.listdir(tmpdir)) == 1

        p.save(os.path.join(tmpdir, "animation.mp4"), save_frames=True)
        assert len(os.listdir(tmpdir)) == 1 + fps * time + 1


@pytest.mark.parametrize(
    "imodule, expected_type", [
        ("panel", PanelAnimation),
        ("ipywidgets", IPYAnimation)
    ]
)
def test_plotgrid_mode_1_different_backends_animation(imodule, expected_type):
    options = dict(show=False, n=10, imodule=imodule)
    p1 = plot(cos(a*x), (x, -3*pi, 3*pi),
        params={a: (1, 5)}, animation={"fps": 10, "time": 1},
        backend=MB, **options)
    p2 = plot(sin(b*x), (x, -3*pi, 3*pi),
        params={b: (2, 4)}, animation={"fps": 4, "time": 2},
        backend=PB, **options)
    p = plotgrid(p1, p2, show=False)
    assert isinstance(p, expected_type)
    assert isinstance(p.backend, PlotGrid)
    # verify that plotgrid collects all parameters from its plots,
    # and that the max fps/time are used
    animation_data = p.animation_data
    fps = 10
    time = 2
    assert animation_data.matrix.shape == (fps * time, 2)
    with TemporaryDirectory(prefix="animation_plogrid_2") as tmpdir:
        with pytest.raises(
            RuntimeError,
            match="Saving plotgrid animation is only supported when"
        ):
            p.save(os.path.join(tmpdir, "animation.gif"))
    # because of the error, one way to test that animation works fine is:
    for i in range(animation_data.matrix.shape[0]):
        p.update_animation(i)


@pytest.mark.parametrize(
    "imodule, expected_type", [
        ("panel", PanelAnimation),
        ("ipywidgets", IPYAnimation)
    ]
)
def test_plotgrid_mode_2_matplotlib(imodule, expected_type):
    params = {
        a: (1, 3),
        b: (0, 1),
        c: (2, pi)
    }
    p1 = plot(
        cos(a*x), params=params, animation={"fps": 5, "time": 0.5},
        show=False, imodule=imodule)
    p2 = plot(
        cos(a*x) * exp(-abs(x) * b), params=params,
        animation={"fps": 6, "time": 0.6}, show=False, imodule=imodule)
    p3 = plot3d(
        cos(x**2 + y**2 - a), prange(x, -c, c), prange(y, -c, c),
        params=params, animation={"fps": 7, "time": 0.7},
        show=False, imodule=imodule)
    gs = GridSpec(2, 2)
    mapping = {
        gs[0, :]: p1,
        gs[1, 0]: p2,
        gs[1, 1]: p3
    }
    p = plotgrid(gs=mapping, show=False)

    assert isinstance(p, expected_type)
    assert isinstance(p.backend, PlotGrid)
    # verify that plotgrid collects all parameters from its plots,
    # and that the max fps/time are used
    animation_data = p.animation_data
    fps = 7
    time = 0.7
    assert animation_data.matrix.shape == (int(fps * time), 3)
    # no error should be raised when saving the animation
    with TemporaryDirectory(prefix="animation_plogrid_1") as tmpdir:
        p.save(os.path.join(tmpdir, "animation.gif"))
        assert len(os.listdir(tmpdir)) == 1

        p.save(os.path.join(tmpdir, "animation.mp4"), save_frames=True)
        assert len(os.listdir(tmpdir)) == 1 + int(fps * time) + 1
