import math
from collections.abc import Iterable
from typing import Self, override

from graph_types import Edge, EdgeAttribute


class NodeBase:
    def __init__(self, parent: Self | None, changes: Iterable[tuple[Edge, EdgeAttribute]], changes_number: int):
        self.parent = parent
        self.changes = tuple(changes)
        # assert changes_number == sum(len(it.changes) for it in self.iterate_to_root())
        self.changes_number = changes_number

    def iterate_to_root(self):
        node = self
        while node is not None:
            yield node
            node = node.parent

    @property
    def encoding(self): return (change for node in self.iterate_to_root() for change in node.changes)


class EvaluatedNode(NodeBase):
    def __init__(self, parent: Self | None, changes: Iterable[tuple[Edge, EdgeAttribute]],
                 changes_number: int, route_distance_violation: float, route_error: float,
                 route: Iterable[Edge] | None):
        super().__init__(parent, changes, changes_number)
        self.route_distance_violation = route_distance_violation
        self.route_error = route_error
        if route is not None:
            self.route: tuple[Edge, ...] | None = tuple(route)
        else:
            self.route = None

    @property
    def feasible(self): return math.isfinite(self.route_distance_violation)
    @property
    def objective(self): return self.route_distance_violation, self.changes_number, self.route_error

    @property
    def processed(self): return self.route is None
    def set_processed(self): self.route = None

    def make_child(self, changes: Iterable[tuple[Edge, EdgeAttribute]],
                   changes_number: int, route_distance_violation: float, route_error: float,
                   route: Iterable[Edge] | None):
        return type(self)(self, changes, changes_number, route_distance_violation, route_error, route)


class Node(EvaluatedNode):
    def __init__(self, parent: Self | None, changes: Iterable[tuple[Edge, EdgeAttribute]],
                 changes_number: int, route_distance_violation: float, route_error: float,
                 route: Iterable[Edge] | None, foil_length: float = 0, fact_length: float | None = 0):
        super().__init__(parent, changes, changes_number, route_distance_violation, route_error, route)
        self.children: dict[tuple[tuple[Edge, EdgeAttribute], ...], Self] = {}
        self.foil_length = foil_length
        self.fact_length = fact_length

    @override
    def make_child(self, changes: Iterable[tuple[Edge, EdgeAttribute]],
                   changes_number: int, route_distance_violation: float, route_error: float,
                   route: Iterable[Edge] | None, foil_length: float = 0, fact_length: float | None = 0):
        child = type(self)(self, changes, changes_number, route_distance_violation, route_error, route,
                           foil_length, fact_length)
        self.children[tuple(changes)] = child
        return child


class LazyNode(NodeBase):
    def evaluate(self, route_distance_violation: float, children: Iterable[Self]):
        self.route_distance_violation = route_distance_violation
        self.children = tuple(children)


class MonteCarloNode(EvaluatedNode):
    class Child:
        def __init__(self, node: 'MonteCarloNode | None' = None):
            self.used = 0
            self.objective = 0.0
            self.node = node

        @property
        def mean_objective(self): return self.objective / self.used if self.used else 0.5

    def __init__(self, parent: Self | None, changes: Iterable[tuple[Edge, EdgeAttribute]],
                 changes_number: int, route_distance_violation: float, route_error: float,
                 route: Iterable[Edge] | None):
        super().__init__(parent, changes, changes_number, route_distance_violation, route_error, route)
        self.children: dict[tuple[tuple[Edge, EdgeAttribute], ...], MonteCarloNode.Child] = {}

    @override
    def make_child(self, changes: Iterable[tuple[Edge, EdgeAttribute]],
                   changes_number: int, route_distance_violation: float, route_error: float,
                   route: Iterable[Edge] | None):
        child = type(self)(self, changes, changes_number, route_distance_violation, route_error, route)
        change_tuple = tuple(changes)
        child_container = self.children.get(change_tuple, MonteCarloNode.Child())
        assert child_container.node is None
        child_container.node = child
        self.children[tuple(changes)] = child_container
        return child
