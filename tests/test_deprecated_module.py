from pytest import warns


def test_deprecation_vectors():
    with warns(
        DeprecationWarning,
        match="`spb.vectors` is deprecated"
    ):
        from spb.vectors import plot_vector


def test_deprecation_functions():
    with warns(
        DeprecationWarning,
        match="`spb.functions` is deprecated"
    ):
        from spb.functions import plot


def test_deprecation_ccomplex_wegert():
    with warns(
        DeprecationWarning,
        match="`spb.ccomplex.wegert` is deprecated"
    ):
        from spb.ccomplex.wegert import wegert


def test_deprecation_ccomplex_complex():
    with warns(
        DeprecationWarning,
        match="`spb.ccomplex.complex` is deprecated"
    ):
        from spb.ccomplex.complex import plot_complex

