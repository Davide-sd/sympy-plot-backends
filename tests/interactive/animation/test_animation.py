import numpy as np
import pytest
from pytest import raises
from spb import plot, MB, BB, PB, KB
from spb.animation import AnimationData
from sympy import symbols, sin, cos, pi, E
from sympy.abc import a, b, c, d, x, y


def test_animation_check_params_keys():
    with raises(
        TypeError,
        match="``params`` must be a dictionary."
    ):
        AnimationData()

    with raises(
        ValueError,
        match="In order to build an animation, at lest one"
    ):
        AnimationData(params={})

    a, b = symbols("a, b")
    params = {a: (0, 5), a + b: (1, 6)}
    with raises(
        ValueError,
        match="All keys of ``params`` must be a single symbol."
    ):
        AnimationData(params=params)


def test_animation_default():
    ani = AnimationData(params={a: (0, 10)})
    assert ani.fps == 30
    assert ani.time == 5
    assert ani.n_frames == 30 * 5


def test_animation_custom():
    fps = 20
    tf = 8
    ani = AnimationData(fps=fps, time=tf, params={a: (0, 10)})
    assert ani.fps == fps
    assert ani.time == tf
    assert ani.n_frames == fps * tf


def test_animation_check_params_values_linear_log_interpolation():
    # linear and log produces different results (useful to simulate a slider)
    p = {a: (1e-03, 1000), b: (1e-03, 1000, "log")}
    ani = AnimationData(params=p)
    assert ani.matrix.shape == (30 * 5, 2)
    assert not np.allclose(ani.matrix[:, 0], ani.matrix[:, 1])
    assert np.allclose(ani.matrix[0, :], 1e-03)
    assert np.allclose(ani.matrix[-1, :], 1000)

    # wrong discretization strategy
    with raises(ValueError, match="Discretization strategy must be either"):
        ani = AnimationData(params={a: (1e-03, 1000, "integers")})

    # symbolic numbers can be used too
    p = {a: (-pi, E)}
    ani = AnimationData(params=p)
    assert ani.matrix.shape == (30 * 5, 1)
    assert np.isclose(ani.matrix[0, 0], -np.pi)
    assert np.isclose(ani.matrix[-1, 0], np.e)


def test_animation_check_params_values_steps():
    # create a parameter with a single step (useful to simulate a checkbox)
    t0, t1, tf = 0, 2.5, 6
    fps = 25
    params = {a: {t0: 0, t1: 1}}
    ani = AnimationData(time=tf, fps=fps, params=params)
    total_frames = fps * tf
    assert ani.matrix.shape == (total_frames, 1)
    frame_idx = int(t1 / tf * total_frames)
    assert np.allclose(ani.matrix[:frame_idx, 0], 0)
    assert np.allclose(ani.matrix[frame_idx:, 0], 1)

    # create a parameter with multiple steps (useful to simulate a dropdown)
    t0, t1, t2, t3, tf = 0, 2, 4, 6, 8
    fps = 20
    params = {a: {t0: 0, t1: 1, t2: 2, t3: 3}}
    ani = AnimationData(time=tf, fps=fps, params=params)
    total_frames = fps * tf
    assert ani.matrix.shape == (total_frames, 1)
    frame_idx1 = int(t1 / tf * total_frames)
    frame_idx2 = int(t2 / tf * total_frames)
    frame_idx3 = int(t3 / tf * total_frames)
    assert (frame_idx1 != frame_idx2) and (frame_idx2 != frame_idx3)
    assert np.allclose(ani.matrix[:frame_idx1, 0], 0)
    assert np.allclose(ani.matrix[frame_idx1:frame_idx2, 0], 1)
    assert np.allclose(ani.matrix[frame_idx2:frame_idx3, 0], 2)
    assert np.allclose(ani.matrix[frame_idx3:, 0], 3)


def test_animation_check_params_values_custom_values():
    # create a parameter with custom values.
    # Here I simulate a ramp-up followed by a ramp-down
    fps = 30
    t1, tf = 3.5, 7
    total_frames = fps * tf
    frame_idx = int(t1 / tf * total_frames)
    v = np.zeros(total_frames)
    v[:frame_idx] = np.linspace(0, 5, frame_idx)
    v[frame_idx:] = 5 - np.linspace(0, 5, frame_idx)
    ani = AnimationData(time=tf, fps=fps, params={a: v})
    assert ani.matrix.shape == (total_frames, 1)
    assert np.allclose(ani.matrix[:, 0], v)

    # create a parameter with custom values.
    # raise error because the number of elements is not equal to the number of
    # frame of the animation
    v = np.cos(np.linspace(0, np.pi, 10))
    with raises(
        ValueError,
        match="The length of the values associated to `a` must"
    ):
        AnimationData(params={a: v})

    # values is not recognized
    with raises(
        TypeError,
        match="The value associated to 'a' is not supported."
    ):
        AnimationData(params={a: b})


def test_animation_get_item():
    fps, tf = 15, 1
    total_frames = fps * tf
    params = {
        a: (0, 5),
        b: {0: -1, 0.5: 1},
        c: np.linspace(1, 2, total_frames)
    }
    ani = AnimationData(fps=fps, time=tf, params=params)
    assert ani.matrix.shape == (total_frames, 3)

    d1 = ani[0]
    assert set(list(d1.keys())) == set([a, b, c])
    assert np.isclose(d1[a], 0)
    assert np.isclose(d1[b], -1)
    assert np.isclose(d1[c], 1)

    d2 = ani[-1]
    assert set(list(d2.keys())) == set([a, b, c])
    assert np.isclose(d2[a], 5)
    assert np.isclose(d2[b], 1)
    assert np.isclose(d2[c], 2)

    d3 = ani[7]
    assert set(list(d3.keys())) == set([a, b, c])
    assert np.isclose(d3[a], 2.5)
    assert np.isclose(d3[b], 1)
    assert np.isclose(d3[c], 1.5)
