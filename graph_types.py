import typing
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Literal

from networkx import MultiGraph

type NodeName = Any
type Edge = tuple[NodeName, NodeName, int]
type EdgeAttribute = Literal['path_type', 'curb_height_max', 'obstacle_free_width_float']
type PathType = Literal['walk', 'bike']


def cast[T](target: type[T], value: Any):
    assert isinstance(value, target)
    return value


class TypedMultiGraph:
    def __init__(self, inner: MultiGraph) -> None:
        self.inner = inner

    @property
    def nodes(self) -> Iterable[NodeName]: return self.inner.nodes
    @property
    def edges(self) -> Iterable[Edge]: return self.inner.edges
    def get_degree(self, node: NodeName): return cast(int, self.inner.degree(node))  # type: ignore
    def get_edge_data(self, edge: Edge): return typing.cast(MutableMapping[str, Any], self.inner.edges[edge])
    def remove_edge(self, edge: Edge): self.inner.remove_edge(*edge)  # type: ignore
    def remove_node(self, node: NodeName): self.inner.remove_node(node)  # type: ignore
    def add_edge(self, edge: Edge, data: Mapping[str, Any]): self.inner.add_edge(*edge, **data)  # type: ignore

    def copy(self): return TypedMultiGraph(cast(MultiGraph, self.inner.copy()))
