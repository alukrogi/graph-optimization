import itertools
from abc import abstractmethod
from collections.abc import Iterable
from typing import override

from graph_types import Edge, EdgeAttribute, PathType, TypedMultiGraph


class ModificationGenerator:
    def __init__(self, graph: TypedMultiGraph, minimum_width: float, maximum_height: float,
                 path_preference: PathType):
        self._graph = graph
        self._width = minimum_width
        self._height = maximum_height
        self._path_type = path_preference
        self._can_narrow = minimum_width > 0.6
        self._can_raise = maximum_height < 0.2

    def get_sensible_modifications(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]],
                                   current_route: Iterable[Edge], edge_name: Edge):
        """Returns a sensible sequence of attributes that need to be changed to modify the edge"""
        edge = self._graph.get_edge_data(edge_name)
        width: float = edge['obstacle_free_width_float']
        height: float | None = edge.get('curb_height_max', None)
        path_type: PathType = edge['path_type']
        can_change_type = path_type == 'walk' or path_type == 'bike'
        initially_feasible = width >= self._width and (height is None or height <= self._height)
        initially_preferred = path_type == self._path_type
        if initially_feasible:
            if any(modified_edge == edge_name for modified_edge,
                   _ in current_solution):
                return ()  # already modified or do not need to make infeasible
            if edge_name in current_route:
                if edge_name in current_route:  # make infeasible
                    if self._can_narrow:
                        return ('obstacle_free_width_float',)
                    if self._can_raise and height is not None:
                        return ('curb_height_max',)
                    if initially_preferred and can_change_type:
                        return ('path_type',)
                    return ()
                elif not initially_preferred and can_change_type:  # make preferred
                    return ('path_type',)
                else:  # nothing to do
                    return ()
            elif not initially_preferred and can_change_type:
                return ('path_type',)
            else:
                return ()
        else:  # make feasible
            changes = [attribute for modified_edge, attribute in current_solution if modified_edge == edge_name]
            need_width = width < self._width and 'obstacle_free_width_float' not in changes
            need_height = height is not None and height > self._height and 'curb_height_max' not in changes
            actions: list[EdgeAttribute] = []
            if need_width:
                actions.append('obstacle_free_width_float')
            if need_height:
                actions.append('curb_height_max')
            if actions:  # need to fix feasibility
                return actions
            # feasibility is fixed but can be made preferred
            return ('path_type',) if not (initially_preferred or 'path_type' in changes) and can_change_type else ()

    def make_preferred(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]], edge_name: Edge):
        edge = self._graph.get_edge_data(edge_name)
        width: float = edge['obstacle_free_width_float']
        height: float | None = edge.get('curb_height_max', None)
        path_type: PathType = edge['path_type']
        can_change_type = path_type == 'walk' or path_type == 'bike'
        initially_preferred = path_type == self._path_type
        changes = [attribute for modified_edge, attribute in current_solution if modified_edge == edge_name]
        need_width = width < self._width and 'obstacle_free_width_float' not in changes
        need_height = height is not None and height > self._height and 'curb_height_max' not in changes
        assert not (need_width or need_height)
        return ('path_type',) if not (initially_preferred or 'path_type' in changes) and can_change_type else ()

    def fix(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]], edge_name: Edge):
        edge = self._graph.get_edge_data(edge_name)
        width: float = edge['obstacle_free_width_float']
        height: float | None = edge.get('curb_height_max', None)
        changes = [attribute for modified_edge, attribute in current_solution if modified_edge == edge_name]
        need_width = width < self._width and 'obstacle_free_width_float' not in changes
        need_height = height is not None and height > self._height and 'curb_height_max' not in changes
        actions: list[EdgeAttribute] = []
        if need_width:
            actions.append('obstacle_free_width_float')
        if need_height:
            actions.append('curb_height_max')
        return actions

    def forbid(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]], edge_name: Edge):
        edge = self._graph.get_edge_data(edge_name)
        width: float = edge['obstacle_free_width_float']
        height: float | None = edge.get('curb_height_max', None)
        path_type: PathType = edge['path_type']
        can_change_type = path_type == 'walk' or path_type == 'bike'
        initially_feasible = width >= self._width and (height is None or height <= self._height)
        initially_preferred = path_type == self._path_type
        assert initially_feasible
        assert not any(modified_edge == edge_name for modified_edge,
                       _ in current_solution)
        if self._can_narrow:
            return ('obstacle_free_width_float',)
        if self._can_raise and height is not None:
            return ('curb_height_max',)
        if initially_preferred and can_change_type:
            return ('path_type',)
        return ()


class ModificationManger(ModificationGenerator):
    @abstractmethod
    def get_modifications(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]], current_route: Iterable[Edge],
                          foil_route: Iterable[Edge]) -> Iterable[Iterable[tuple[Edge, EdgeAttribute]]]: ...


class HeuristicModifications(ModificationManger):
    @override
    def get_modifications(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]],
                          current_route: Iterable[Edge], foil_route: Iterable[Edge]):
        breaks = filter(lambda x: x,
                        (put_edge(edge,
                                  self.get_sensible_modifications(current_solution, current_route, edge))
                         for edge in current_route))
        fixes = filter(lambda x: x,
                       (put_edge(edge,
                                 self.get_sensible_modifications(current_solution, current_route, edge))
                        for edge in foil_route))
        return itertools.chain(breaks, fixes)


class AllModifications(ModificationManger):
    @override
    def get_modifications(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]],
                          current_route: Iterable[Edge], foil_route: Iterable[Edge]):
        return filter(lambda x: x,
                      (put_edge(edge,
                                self.get_sensible_modifications(current_solution, current_route, edge))
                       for edge in self._graph.edges))


class FastModifications(ModificationManger):
    @override
    def get_modifications(self, current_solution: Iterable[tuple[Edge, EdgeAttribute]],
                          current_route: Iterable[Edge], foil_route: Iterable[Edge]):
        if not current_solution:
            initial = [action for edge in foil_route for action in put_edge(edge, self.fix(current_solution, edge))]
            if initial:
                return (initial,)
        breaks = (put_edge(edge,
                           self.get_sensible_modifications(current_solution, current_route, edge))
                  for edge in frozenset(current_route).difference(foil_route))
        fixes = (put_edge(edge,
                          self.make_preferred(current_solution, edge))
                 for edge in frozenset(foil_route).difference(current_route))
        return filter(lambda x: x, itertools.chain(breaks, fixes))


def put_edge(edge: Edge, modifications: Iterable[EdgeAttribute]) -> list[tuple[Edge, EdgeAttribute]]:
    return [(edge, attribute) for attribute in modifications]
