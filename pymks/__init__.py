"""PyMKS - the materials knowledge system in Python

See the documenation for details at https://pymks.org
"""

try:
    import pyfftw  # pylint: disable=unused-import; # noqa: F401

    # ensure that pyfftw is always imported before numpy to avoid
    # https://github.com/materialsinnovation/pymks/issues/304
except ImportError:
    pass

import os

from ._version import get_versions

from .fmks.data.cahn_hilliard import solve_cahn_hilliard
from .fmks.plot import plot_microstructures
from .fmks.bases.primitive import PrimitiveTransformer
from .fmks.bases.legendre import LegendreTransformer
from .fmks.localization import LocalizationRegressor
from .fmks.localization import ReshapeTransformer
from .fmks.localization import coeff_to_real
from .fmks.data.delta import generate_delta
from .fmks.data.multiphase import generate_multiphase
from .fmks.correlations import FlattenTransformer
from .fmks.correlations import TwoPointCorrelation
from .fmks.data.checkerboard import generate_checkerboard
from .fmks.pair_correlations import paircorr_from_twopoint
from .fmks.correlations import two_point_stats

try:
    import sfepy  # noqa: F401
except ImportError:

    def solve_fe(*_, **__):
        """Dummy funcion when sfepy unavailable
        """
        # pylint: disable=redefined-outer-name, import-outside-toplevel, unused-import
        import sfepy  # noqa: F401, F811


else:
    from .fmks.data.elastic_fe import solve_fe

# the following will be deprecated
from .mks_localization_model import MKSLocalizationModel
from .bases.primitive import PrimitiveBasis
from .bases.legendre import LegendreBasis
from .mks_structure_analysis import MKSStructureAnalysis
from .mks_homogenization_model import MKSHomogenizationModel

MKSRegressionModel = MKSLocalizationModel
DiscreteIndicatorBasis = PrimitiveBasis
ContinuousIndicatorBasis = PrimitiveBasis
# the above will be deprecatec


def test():
    r"""
    Run all the doctests available.
    """
    import pytest  # pylint: disable=import-outside-toplevel

    path = os.path.join(os.path.split(__file__)[0], "fmks")
    pytest.main(args=[path, "--doctest-modules", "-r s"])


__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "__version__",
    "test",
    "solve_cahn_hilliard",
    "plot_microstructures",
    "PrimitiveTransformer",
    "LocalizationRegressor",
    "ReshapeTransformer",
    "coeff_to_real",
    "MKSLocalizationModel",
    "PrimitiveBasis",
    "LegendreBasis",
    "MKSHomogenizationModel",
    "MKSStructureAnalysis",
    "generate_delta",
    "LegendreTransformer",
    "solve_fe",
    "generate_multiphase",
    "FlattenTransformer",
    "TwoPointCorrelation",
    "generate_checkerboard",
    "paircorr_from_twopoint",
    "two_point_stats",
]
