import itertools
import typing
from collections.abc import Iterable, Mapping
from typing import Any

from networkx import Graph

from graph_types import Edge, EdgeAttribute, PathType
from utility import Modification


class ModificationGenerator:
    def __init__(self, graph: Graph, minimum_width: float, maximum_height: float,
                 path_preference: PathType):
        self._graph = graph
        self._width = minimum_width
        self._height = maximum_height
        self._path_type = path_preference
        self._can_narrow = minimum_width > 0.6
        self._can_raise = maximum_height < 0.2

    def get_sensible_modifications(self, current_solution: Iterable[Modification],
                                   current_route: Iterable[Edge], edge_name: Edge, foil_route):
        """Returns a sensible sequence of attributes that need to be changed to modify the edge"""
        edge = typing.cast(Mapping[EdgeAttribute, Any], self._graph.edges[edge_name])
        width: float = edge['obstacle_free_width_float']
        height: float | None = edge.get('curb_height_max', None)
        path_type: PathType = edge['path_type']
        can_change_type = path_type == 'walk' or path_type == 'bike'
        initially_feasible = width >= self._width and (height is None or height <= self._height)
        initially_preferred = path_type == self._path_type
        if initially_feasible:
            if any(modified_edge == edge_name for modified_edge, _ in current_solution):  # already modified
                return ()
            if edge_name in current_route:  # make infeasible
                if self._can_narrow:
                    # print('width')
                    return ('obstacle_free_width_float',)
                if self._can_raise and height is not None:
                    # print('curb')
                    return ('curb_height_max',)
                if initially_preferred and can_change_type:
                    # print('path type cancel')
                    return ('path_type',)
                return ()
            elif not initially_preferred and can_change_type:  # make preferred
                # print('path type add')
                return ('path_type',)
            else:  # nothing to do
                return ()
        else:  # make feasible
            changes = [attribute for edge, attribute in current_solution if edge == edge_name]
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

    def forbid(self, edge_name: Edge):
        edge = self._graph.get_edge_data(edge_name)
        width: float = edge['obstacle_free_width_float']
        height: float | None = edge.get('curb_height_max', None)
        path_type: PathType = edge['path_type']
        can_change_type = path_type == 'walk' or path_type == 'bike'
        initially_feasible = width >= self._width and (height is None or height <= self._height)
        initially_preferred = path_type == self._path_type
        if not initially_feasible:
            return ()
        if self._can_narrow:
            return ('obstacle_free_width_float',)
        if self._can_raise and height is not None:
            return ('curb_height_max',)
        if initially_preferred and can_change_type:
            return ('path_type',)
        return ()


class HeuristicModifications(ModificationGenerator):
    def get_modifications(self, current_solution: Iterable[Modification],
                          current_route: Iterable[Edge], foil_route: Iterable[Edge], swapping: bool, true_foil=0):
        if true_foil == 0:
            true_foil = foil_route
        if swapping:
            breaks = filter(lambda x: x,
                            (_put_edge(edge, self.get_sensible_modifications(current_solution, current_route, edge,
                                                                             foil_route))
                             for edge in current_route if edge not in foil_route))
            fixes = filter(lambda x: x,
                           (_put_edge(edge, self.get_sensible_modifications(current_solution, current_route, edge,
                                                                            foil_route))
                            for edge in []))
        else:
            breaks = filter(lambda x: x,
                            (_put_edge(edge, self.get_sensible_modifications(current_solution, current_route, edge,
                                                                             foil_route))
                             for edge in current_route if edge not in true_foil))
            fixes = filter(lambda x: x,
                           (_put_edge(edge, self.get_sensible_modifications(current_solution, current_route, edge,
                                                                            foil_route))
                            for edge in foil_route if edge not in current_route))
        return itertools.chain(breaks, fixes)


def _put_edge(edge: Edge, modifications: Iterable[EdgeAttribute]) -> tuple[Modification, ...]:
    return tuple((edge, attribute) for attribute in modifications)
