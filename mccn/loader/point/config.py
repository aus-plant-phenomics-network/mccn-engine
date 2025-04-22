from dataclasses import dataclass

from mccn._types import MergeMethods


@dataclass
class PointLoadConfig:
    """Point load config - determines how point data should be aggregated and interpolated"""

    agg_method: MergeMethods = "mean"
    """Merge method for aggregation"""
