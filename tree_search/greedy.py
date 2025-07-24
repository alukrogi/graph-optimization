import math
import random
import typing
from collections.abc import Iterable, Mapping, Sequence

from graph_types import Edge, EdgeAttribute, TypedMultiGraph
from objective import find_path, find_path_fast, route_difference
from tree_search import modification
from tree_search.modification import ModificationManger
from tree_search.tree import MonteCarloNode, Node
from utility import Solution, UserModel


def random_fix(modification_manager: ModificationManger, graph: TypedMultiGraph, foil_route: Sequence[Edge],
               user_model: UserModel, delta: float, initial_node: Node, weight_mapping: Mapping[Edge, float],
               depth_limit: int):
    node = initial_node
    for edge in foil_route:
        if node.route_distance_violation == 0 or node.changes_number > depth_limit:
            return node
        modifications = modification_manager.fix(node.encoding, edge)
        if modifications:
            modifications = tuple(modification.put_edge(edge, modifications))
            node = add_modification(graph, foil_route, user_model, delta, node, modifications)
    encoding = list(node.encoding)
    while len(encoding) < depth_limit:
        route = node.route
        if route is None or node.route_distance_violation == 0:
            break
        modifications = list(modification_manager.get_modifications(encoding, route, foil_route))
        weights = [sum(weight_mapping.get(edge, 1.0) for edge, _ in change_set) for change_set in modifications]
        choice = tuple(random.choices(modifications, weights)[0])
        node = add_modification(graph, foil_route, user_model, delta, node, choice)
        encoding = list(node.encoding)
    return node


@typing.overload
def fix_first(modification_manager: ModificationManger, graph: TypedMultiGraph, foil_route: Sequence[Edge],
              user_model: UserModel, delta: float, initial_node: Node, depth_limit: int) -> Node: ...


@typing.overload
def fix_first(modification_manager: ModificationManger, graph: TypedMultiGraph, foil_route: Sequence[Edge],
              user_model: UserModel, delta: float, initial_node: MonteCarloNode,
              depth_limit: int) -> MonteCarloNode: ...


def fix_first(modification_manager: ModificationManger, graph: TypedMultiGraph, foil_route: Sequence[Edge],
              user_model: UserModel, delta: float, initial_node: Node | MonteCarloNode, depth_limit: int):
    node = initial_node
    for edge in foil_route:
        if node.route_distance_violation == 0 or node.changes_number > depth_limit:
            return node
        modifications = modification_manager.fix(node.encoding, edge)
        if modifications:
            modifications = tuple(modification.put_edge(edge, modifications))
            node = add_modification(graph, foil_route, user_model, delta, node, modifications)
    for edge in foil_route:
        if node.route_distance_violation == 0 or node.changes_number > depth_limit:
            return node
        modifications = modification_manager.make_preferred(node.encoding, edge)
        if modifications:
            modifications = tuple(modification.put_edge(edge, modifications))
            new_node = add_modification(graph, foil_route, user_model, delta, node, modifications)
            if new_node.objective < node.objective:
                node = new_node
    while node.changes_number < depth_limit and node.route_distance_violation != 0:
        edge = None
        route = node.route
        if route is None:
            return node
        for foil, fact in zip(foil_route, route):
            if fact != foil:
                edge = fact
                break
        assert edge is not None
        modifications = modification_manager.forbid(node.encoding, edge)
        if len(modificiations) == 0:
            break
        modifications = tuple(modification.put_edge(edge, modifications))
        node = add_modification(graph, foil_route, user_model, delta, node, modifications)
    return node


# def length_fix(instance: Instance, modification_manager: ModificationManger, initial_node: Node, depth_limit: int):
#     node=initial_node
#     for edge in instance.foil_route:
#         if node.route_distance_violation == 0 or node.changes_number > depth_limit:
#             return node
#         modifications = modification_manager.fix(node.encoding, edge)
#         if modifications:
#             modifications = tuple(modification.put_edge(edge, modifications))
#             node = add_modification(*instance, node, modifications)
#     while node.changes_number < depth_limit and node.route_distance_violation != 0:
#         edge = None
#         route = node.route
#         if route is None:
#             return node
#         best


def try_reduce(graph: TypedMultiGraph, foil_route: Sequence[Edge],
               user_model: UserModel, delta: float, encoding: Iterable[tuple[Edge, EdgeAttribute]], objective: float):
    begin = foil_route[0][0]
    end = foil_route[-1][1]
    encoding = list(encoding)
    i = 0
    while i < len(encoding):
        copied = encoding.copy()
        del copied[i]
        route = find_path_fast(graph, user_model, begin, end, copied)
        if route is None:
            i += 1
            continue
        distance = route_difference(foil_route, route, graph)
        if distance - delta > objective:
            i += 1
            continue
        encoding = copied
    route = find_path(graph, encoding, user_model, begin, end)
    assert route is not None
    distance = route_difference(foil_route, route, graph)
    return Solution(encoding, (max(distance - delta, 0), len(encoding), distance))


def add_modification(graph: TypedMultiGraph, foil_route: Sequence[Edge], user_model: UserModel, delta: float,
                     node: Node | MonteCarloNode, modification: tuple[tuple[Edge, EdgeAttribute], ...]):
    if modification in node.children:
        if isinstance(node, Node):
            return node.children[modification]
        else:
            typing.assert_type(node, MonteCarloNode)
            ret = node.children[modification].node
            if ret is not None:
                return ret
    begin = foil_route[0][0]
    end = foil_route[-1][1]
    encoding = list(node.encoding)
    encoding.extend(modification)
    route = find_path_fast(graph, user_model, begin, end, encoding)
    if route is not None:
        distance = route_difference(foil_route, route, graph)
    else:
        distance = math.inf
    new_node = node.make_child(modification, len(encoding), max(distance - delta, 0), distance, route)
    return new_node
