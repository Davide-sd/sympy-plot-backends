import pytest
from spb.defaults import cfg


@pytest.fixture
def ipywidgets_options():
    return dict(show=False, imodule="ipywidgets")


@pytest.fixture
def panel_options():
    return dict(show=False, imodule="panel")


@pytest.fixture
def default_range():
    return lambda s: (s, cfg["plot_range"]["min"], cfg["plot_range"]["max"])


@pytest.fixture
def default_complex_range():
    _min = cfg["plot_range"]["min"]
    _max = cfg["plot_range"]["max"]
    return lambda s: (s, _min + _min * 1j, _max + _max * 1j)
