from dataclasses import dataclass

from mccn._types import MergeMethod_Map_T


@dataclass
class PointLoadConfig:
    """Point load config - determines how point data should be aggregated and interpolated"""

    agg_method: MergeMethod_Map_T = "mean"
    """Merge method for aggregation"""
