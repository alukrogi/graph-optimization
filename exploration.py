import functools
import heapq
from collections.abc import Iterable
from typing import Self

from tree import Node


class SimpleExploration:
    def __init__(self, depth_first: bool = False):
        self._queue: list[Node] = []
        self._depth_first = depth_first

    def add_node(self, item: Node):
        if item.route_distance_violation != 0:
            self._queue.append(item)

    def add_nodes(self, items: Iterable[Node]): self._queue.extend(
        item for item in items if item.route_distance_violation > 0 and item.feasible)

    def has_next(self): return bool(self._queue)
    @property
    def pending(self): return len(self._queue)

    def pop_next(self): return self._queue.pop(-1 if self._depth_first else 0)


class GreedyExploration:
    @functools.total_ordering
    class _NodeWrapper:
        def __init__(self, node: Node):
            self.node = node

        def __eq__(self, value: object):
            if not isinstance(value, GreedyExploration._NodeWrapper):
                return False
            return self.node.objective == value.node.objective

        def __lt__(self, other: Self):
            return self.node.objective < other.node.objective

    def __init__(self):
        self._queue: list[GreedyExploration._NodeWrapper] = []

    def add_node(self, item: Node):
        if item.route_distance_violation != 0 and item.feasible:
            heapq.heappush(self._queue, GreedyExploration._NodeWrapper(item))

    def add_nodes(self, items: Iterable[Node]):
        self._queue.extend(GreedyExploration._NodeWrapper(item) for item in items if item.route_distance_violation > 0)
        heapq.heapify(self._queue)

    def has_next(self): return bool(self._queue)
    @property
    def pending(self): return len(self._queue)

    def pop_next(self): return heapq.heappop(self._queue).node


class LengthExploration:
    @functools.total_ordering
    class _NodeWrapper:
        def __init__(self, node: Node):
            self.node = node
            fact_length = node.fact_length
            assert fact_length is not None
            self.objective = (node.foil_length - fact_length, node.changes_number, node.route_error)

        def __eq__(self, value: object):
            if not isinstance(value, LengthExploration._NodeWrapper):
                return False
            return self.objective == value.objective

        def __lt__(self, other: Self):
            return self.objective < other.objective

    def __init__(self):
        self._queue: list[LengthExploration._NodeWrapper] = []

    def add_node(self, item: Node):
        if item.route_distance_violation != 0 and item.feasible:
            heapq.heappush(self._queue, LengthExploration._NodeWrapper(item))

    def add_nodes(self, items: Iterable[Node]):
        self._queue.extend(LengthExploration._NodeWrapper(item) for item in items if item.route_distance_violation > 0)
        heapq.heapify(self._queue)

    def has_next(self): return bool(self._queue)
    @property
    def pending(self): return len(self._queue)

    def pop_next(self): return heapq.heappop(self._queue).node
