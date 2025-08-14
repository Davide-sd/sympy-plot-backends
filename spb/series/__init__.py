from spb.series.base import (
    BaseSeries
)
from spb.series.complex_analysis import (
    ComplexPointSeries,
    ComplexSurfaceBaseSeries,
    ComplexDomainColoringBaseSeries,
    ComplexSurfaceSeries,
    ComplexDomainColoringSeries,
    RiemannSphereSeries,
    ComplexParametric3DLineSeries,
    AbsArgLineSeries,
)
from spb.series.vectors import (
    Vector2DSeries,
    Vector3DSeries,
    SliceVector3DSeries,
    Arrow2DSeries,
    Arrow3DSeries
)
from spb.series.control import (
    SGridLineSeries,
    ZGridLineSeries,
    NicholsLineSeries,
    NyquistLineSeries,
    RootLocusSeries,
    SystemResponseSeries,
    ColoredSystemResponseSeries,
    PoleZeroWithSympySeries,
    PoleZeroSeries,
    NGridLineSeries,
    MCirclesSeries
)
from spb.series.series_2d_3d import (
    List2DSeries,
    List3DSeries,
    LineOver1DRangeSeries,
    ColoredLineOver1DRangeSeries,
    Parametric2DLineSeries,
    Parametric3DLineSeries,
    ContourSeries,
    SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries,
    ImplicitSeries,
    Implicit3DSeries,
    PlaneSeries,
    Geometry2DSeries,
    Geometry3DSeries,
    GenericDataSeries,
    HVLineSeries
)
from spb.doc_utils.ipython import generate_doc


series = [
    BaseSeries,
    ComplexPointSeries,
    ComplexSurfaceBaseSeries,
    ComplexDomainColoringBaseSeries,
    ComplexSurfaceSeries,
    ComplexDomainColoringSeries,
    RiemannSphereSeries,
    ComplexParametric3DLineSeries,
    AbsArgLineSeries,
    Vector2DSeries,
    Vector3DSeries,
    SliceVector3DSeries,
    Arrow2DSeries,
    Arrow3DSeries,
    SGridLineSeries,
    ZGridLineSeries,
    NicholsLineSeries,
    NyquistLineSeries,
    RootLocusSeries,
    SystemResponseSeries,
    ColoredSystemResponseSeries,
    PoleZeroWithSympySeries,
    PoleZeroSeries,
    NGridLineSeries,
    MCirclesSeries,
    List2DSeries,
    List3DSeries,
    LineOver1DRangeSeries,
    ColoredLineOver1DRangeSeries,
    Parametric2DLineSeries,
    Parametric3DLineSeries,
    ContourSeries,
    SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries,
    ImplicitSeries,
    Implicit3DSeries,
    PlaneSeries,
    Geometry2DSeries,
    Geometry3DSeries,
    GenericDataSeries,
    HVLineSeries
]

generate_doc(*series)

__all__ = [s.__name__ for s in series]
