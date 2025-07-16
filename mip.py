import itertools
from collections.abc import Sequence
from enum import Enum, auto
from typing import Literal

import pulp
import pyscipopt
import pyscipopt.scip
from pulp import LpProblem

from objective import find_path_fast, compute_objective
from graph_types import Edge, EdgeAttribute, PathType, TypedMultiGraph
from utility import Instance, Modification, UserModel


class EdgeState(Enum):
    SHORT = auto()
    LONG = auto()
    ABSENT = auto()


def calculate_change_costs(graph: TypedMultiGraph, user_model: UserModel):
    result: dict[tuple[Edge, EdgeState], int] = {}
    for edge in graph.edges:
        data = graph.get_edge_data(edge)
        width: float = data['obstacle_free_width_float']
        height: float | None = data.get('curb_height_max', None)
        path_type: PathType = data['path_type']
        width_infeasible = width < user_model.minimum_width
        height_infeasible = height is not None and height > user_model.maximum_height
        preferred = path_type == user_model.path_preference
        make_feasible = width_infeasible + height_infeasible
        result[edge, EdgeState.ABSENT] = 1 - (width_infeasible or height_infeasible)
        result[edge, EdgeState.SHORT] = make_feasible + 1 - preferred
        result[edge, EdgeState.LONG] = make_feasible + preferred
    return result


def calculate_changes(graph: TypedMultiGraph, user_model: UserModel):
    result: dict[tuple[Edge, EdgeState], Sequence[EdgeAttribute]] = {}
    for edge in graph.edges:
        data = graph.get_edge_data(edge)
        width: float = data['obstacle_free_width_float']
        height: float | None = data.get('curb_height_max', None)
        path_type: PathType = data['path_type']
        width_infeasible = width < user_model.minimum_width
        height_infeasible = height is not None and height > user_model.maximum_height
        preferred = path_type == user_model.path_preference
        make_feasible: list[EdgeAttribute] = []
        if width_infeasible:
            make_feasible.append('obstacle_free_width_float')
        if height_infeasible:
            make_feasible.append('curb_height_max')
        if width_infeasible or height_infeasible:
            result[edge, EdgeState.ABSENT] = ()
        else:
            result[edge, EdgeState.ABSENT] = ('obstacle_free_width_float',)
        result[edge, EdgeState.SHORT] = make_feasible + ([] if preferred else ['path_type'])
        result[edge, EdgeState.LONG] = make_feasible + ([] if not preferred else ['path_type'])
    return result


def calculate_edge_weights(graph: TypedMultiGraph, user_model: UserModel):
    result: dict[tuple[Edge, Literal[EdgeState.SHORT, EdgeState.LONG]], float] = {}
    for edge in graph.edges:
        data = graph.get_edge_data(edge)
        initial: float = data['my_weight']
        path_type: PathType = data['path_type']
        preferred = path_type == user_model.path_preference
        if preferred:
            result[edge, EdgeState.SHORT] = initial
            result[edge, EdgeState.LONG] = initial / user_model.preference_weight
        else:
            result[edge, EdgeState.SHORT] = initial * user_model.preference_weight
            result[edge, EdgeState.LONG] = initial
    return result


def build_two_level_model(instance: Instance) -> LpProblem:
    model = LpProblem()
    nodes = list(instance.graph.nodes)
    x = pulp.LpVariable.dicts('x', itertools.product(instance.graph.edges, [EdgeState.SHORT, EdgeState.LONG]),
                              cat=pulp.LpBinary)
    t = pulp.LpVariable.dicts('t', itertools.product(instance.graph.edges, [EdgeState.SHORT, EdgeState.LONG]), 0, 1)
    b = pulp.LpVariable.dicts('b', itertools.product(instance.graph.edges, [EdgeState.SHORT, EdgeState.LONG]), 0)
    a = pulp.LpVariable.dicts('a', nodes)

    change_costs = calculate_change_costs(instance.graph, instance.user_model)
    obj = 0
    for (edge, state), cost in change_costs.items():
        match state:
            case EdgeState.SHORT | EdgeState.LONG:
                obj += cost * x[edge, state]
            case EdgeState.ABSENT:
                obj += cost * (1 - x[edge, EdgeState.SHORT] - x[edge, EdgeState.LONG])
    model += obj
    for edge in instance.graph.edges:
        model += x[edge, EdgeState.SHORT] + x[edge, EdgeState.LONG] <= 1
        for state in EdgeState.SHORT, EdgeState.LONG:
            model += t[edge, state] <= x[edge, state]
    foil_length = sum(instance.graph.get_edge_data(edge)['length'] for edge in instance.foil_route)
    common_length = pulp.lpSum(instance.graph.get_edge_data(edge)['length'] * t[edge, state]
                               for edge in instance.foil_route for state in (EdgeState.SHORT, EdgeState.LONG))
    fact_length = pulp.lpSum(instance.graph.get_edge_data(edge)['length'] * t[edge, state]
                             for edge in instance.graph.edges for state in (EdgeState.SHORT, EdgeState.LONG))
    model += (foil_length + fact_length) * (1 - instance.delta) <= 2 * common_length
    origin = instance.foil_route[0][0]
    destination = instance.foil_route[-1][1]
    for node in nodes:
        if node == origin:
            balance = -1
        elif node == destination:
            balance = 1
        else:
            balance = 0
        ingoing = pulp.lpSum(t[edge, state] for edge in instance.graph.edges if node == edge[1]
                             for state in (EdgeState.SHORT, EdgeState.LONG))
        outgoing = pulp.lpSum(t[edge, state] for edge in instance.graph.edges if node == edge[0]
                              for state in (EdgeState.SHORT, EdgeState.LONG))
        model += ingoing - outgoing == balance
    weights = calculate_edge_weights(instance.graph, instance.user_model)
    for (i, j, n), k in b:
        model += a[j] - a[i] - b[(i, j, n), k] <= weights[(i, j, n), k]
    m = pulp.LpVariable.dicts('m', x, 0)
    model += pulp.lpSum(t[key] * weights[key] for key in t) <= -a[origin] + a[destination] - pulp.lpSum(m.values())
    M = 2 * foil_length / instance.user_model.preference_weight
    for key in x:
        model += m[key] <= b[key]
        model += m[key] <= x[key] * M
        model += m[key] >= b[key] - (1 - x[key]) * M
    return model, x


def get_encoding(graph: TypedMultiGraph, user_model: UserModel, x):
    all_changes = calculate_changes(graph, user_model)
    result: list[Modification] = []
    for edge in graph.edges:
        for state in EdgeState.SHORT, EdgeState.LONG:
            if x[edge, state].value() > 0.5:
                target = state
                break
        else:
            target = EdgeState.ABSENT
        changes = all_changes[edge, target]
        result.extend((edge, modification) for modification in changes)
    return result


def solve_mip(instance: Instance):
    opt = [f'{param}=60'for param in pyscipopt.Model().getParams().keys() if 'presoltiming' in param]
    model, x = build_two_level_model(instance)
    print(model.objective.constant)
    # model.solve(pulp.SCIP_PY(timeLimit=2 * 60, msg=True, options=['numerics/feastol=1e-9']))
    model.solve(pulp.SCIP_PY(timeLimit=15 * 60, msg=True, options=opt))
    print(model.objective.value(), model.solverModel.getDualbound() + model.objective.constant)
    encoding = get_encoding(instance.graph, instance.user_model, x)
    begin, end = instance.endpoints
    route = find_path_fast(instance.graph, instance.user_model, begin, end, encoding)
    objective = compute_objective(instance, encoding, route)
    return objective
