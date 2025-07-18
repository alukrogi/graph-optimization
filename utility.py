import itertools
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple, Optional, Any

from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry

from graph_types import Edge, EdgeAttribute, NodeName, PathType, TypedMultiGraph


@dataclass(frozen=True)
class UserModel:
    minimum_width: float
    maximum_height: float
    path_preference: PathType
    preference_weight: float


class Instance(NamedTuple):
    graph: TypedMultiGraph
    foil_route: tuple[Edge, ...]
    user_model: UserModel
    delta: float

    @property
    def endpoints(self) -> tuple[NodeName, NodeName]:
        """
        The start and end nodes of the foil route.

        Raises:
            ValueError: if `foil_route` is empty.
        """
        if not self.foil_route:
            raise ValueError("Cannot compute endpoints of an empty foil_route")
        first_edge, *_, last_edge = self.foil_route
        start_node = first_edge[0]
        end_node = last_edge[1]
        return start_node, end_node


class FullInstanceData(NamedTuple):
    instance: Instance
    gdf_coords: GeoDataFrame
    origin_node_loc: BaseGeometry
    dest_node_loc: BaseGeometry
    meta_data: dict[str, Any]
    df: GeoDataFrame
    df_path_foil: GeoDataFrame


type Modification = tuple[Edge, EdgeAttribute]
type Objective = tuple[float, int, float]


class Solution:
    """
    A candidate solution in LNS/ALNS:
      - encoding: the list of modifications applied
      - objective: (route error (violation), graph error (num_changes), route_diff)
      - route: the actual edge‐route found under these encodings
    """

    def __init__(
            self,
            encoding: Iterable[Modification],
            objective: Objective,
            route: Optional[Iterable[Edge]] = None
    ) -> None:
        self.encoding: tuple[Modification, ...] = tuple(encoding)
        self.objective: Objective = objective
        self.route: Optional[tuple[Edge, ...]] = None if route is None else tuple(route)

    def __copy__(self) -> "Solution":
        return Solution(self.encoding, self.objective, self.route)


class Timer:
    """Monotonic stopwatch with a deadline."""

    def __init__(self, time_limit: float, time_to_best: float = 0.0):
        self.start = time.monotonic()
        self.deadline = self.start + time_limit
        self._last_log = self.start
        self.time_to_best = time_to_best

    def out(self) -> bool:
        return time.monotonic() >= self.deadline

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def postpone(self, time_shift: float):
        self.deadline = self.deadline + time_shift

    def update_time_to_best(self):
        self.time_to_best = self.elapsed()

    def set_time_to_best(self, value: float):
        self.time_to_best = value

    def get_time_to_best(self) -> float:
        return self.time_to_best

    def should_log(self, period: float) -> bool:
        now = time.monotonic()
        if now - self._last_log >= period:
            self._last_log = now
            return True
        return False


ABSENT_COEFFICIENT = 2 ** 16


def node_path_to_edges(path: Iterable[NodeName]):
    return [(u, v, 0) for u, v in itertools.pairwise(path)]

def calculate_segment_time_limit(total_time_limit: float):
    if total_time_limit < 0:
        return 0.0
    # three segments + space for overhead
    return ((total_time_limit - 5) if total_time_limit < 30 else (total_time_limit * 0.99 - 10)) / 3.0
