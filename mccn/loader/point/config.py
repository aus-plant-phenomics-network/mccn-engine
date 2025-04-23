from dataclasses import dataclass

from mccn._types import Number_T


@dataclass
class PointLoadConfig:
    """Point load config - determines how point data should be aggregated and interpolated"""

    radius: Number_T = 0.5
