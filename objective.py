from collections.abc import Iterable, Mapping
from typing import Any, Literal, no_type_check, overload

from geopandas import GeoDataFrame

import utility
from graph_types import Edge, NodeName, TypedMultiGraph
from router import Router
from utility import Instance, Modification, UserModel, Objective
from validation.dataparser import handle_weight_with_recovery
from validation.graph_op import pertub_with_op_list, graphOperator


def create_output(df: GeoDataFrame, graph: TypedMultiGraph, metadata_user_model,
                  encoding: Iterable[Modification]):
    maximum_height = metadata_user_model["max_curb_height"]
    minimum_width = metadata_user_model["min_sidewalk_width"]
    op_list: list[tuple[str, tuple[int, Any], float | str, Literal['success']]] = []
    for edge, attribute in encoding:
        edge_attrs = graph.get_edge_data(edge)
        row = df.iloc[edge_attrs['index_position']]
        if attribute == 'path_type':
            current_path_type = edge_attrs['path_type']
            if current_path_type == 'walk':
                op_list.append(
                    ('modify_path_type', (edge_attrs['index_position'], row.geometry), 'bike', 'success'))
            else:
                op_list.append(
                    ('modify_path_type', (edge_attrs['index_position'], row.geometry), 'walk', 'success'))
        elif attribute == 'curb_height_max':
            if edge_attrs['curb_height_max'] > maximum_height:
                op_list.append(('sub_curb_height', (edge_attrs['index_position'], row.geometry),
                                edge_attrs['curb_height_max'], 'success'))
            else:
                op_list.append(('add_curb_height', (edge_attrs['index_position'], row.geometry),
                                0.2 - edge_attrs['curb_height_max'], 'success'))
        elif attribute == 'obstacle_free_width_float':
            if edge_attrs['obstacle_free_width_float'] < minimum_width:
                op_list.append(('add_width', (edge_attrs['index_position'], row.geometry),
                                2.0 - edge_attrs['obstacle_free_width_float'], 'success'))
            else:
                op_list.append(('sub_width', (edge_attrs['index_position'], row.geometry),
                                edge_attrs['obstacle_free_width_float'] - 0.6, 'success'))

    graph_operator = graphOperator()
    df_p, result_op_list = pertub_with_op_list(graph_operator, op_list, df)
    result_df = handle_weight_with_recovery(df_p, metadata_user_model)
    return result_df, result_op_list


@no_type_check
def create_output2(dataframe: GeoDataFrame, graph: TypedMultiGraph, user_model: UserModel,
                   encoding: Iterable[Modification]):
    result_df = dataframe.copy()
    result: list[tuple[str, tuple[int, Any], float | str, Literal['success']]] = []
    for edge, attribute in encoding:
        edge_attrs = graph.get_edge_data(edge)
        row = dataframe.iloc[edge_attrs['index_position']]
        row_id = edge_attrs['index_position']
        if attribute == 'path_type':
            current_path_type = edge_attrs['path_type']
            weight_coeff = 1.0 / user_model.preference_weight if (
                    user_model.path_preference == current_path_type) else user_model.preference_weight
            if current_path_type == 'walk':
                result_df.at[row_id, 'path_type'] = 'bike'
                result_df.at[row_id, 'my_weight'] = result_df.at[row_id, 'my_weight'] * weight_coeff
                result.append(('modify_path_type', (edge_attrs['index_position'], row.geometry), 'bike', 'success'))
            else:
                result.append(('modify_path_type', (edge_attrs['index_position'], row.geometry), 'walk', 'success'))
                result_df.at[row_id, 'path_type'] = 'walk'
                result_df.at[row_id, 'my_weight'] = result_df.at[row_id, 'my_weight'] * weight_coeff
        elif attribute == 'curb_height_max':
            if edge_attrs['curb_height_max'] > user_model.maximum_height:
                if result_df.at[row_id, 'include'] == 0:
                    if 'obstacle_free_width_float' in edge_attrs.keys():
                        if edge_attrs['obstacle_free_width_float'] is not None:
                            if edge_attrs['obstacle_free_width_float'] >= user_model.minimum_width:
                                result_df.at[row_id, 'include'] = 1
                        else:
                            result_df.at[row_id, 'include'] = 1
                    else:
                        result_df.at[row_id, 'include'] = 1
                result_df.at[row_id, 'curb_height_max'] = 0.0
                result.append(('sub_curb_height', (edge_attrs['index_position'], row.geometry),
                               edge_attrs['curb_height_max'], 'success'))
            else:
                if result_df.at[row_id, 'include'] == 1:
                    result_df.at[row_id, 'include'] = 0
                result_df.at[row_id, 'curb_height_max'] = 0.2
                result.append(('add_curb_height', (edge_attrs['index_position'], row.geometry),
                               0.2 - edge_attrs['curb_height_max'], 'success'))
        elif attribute == 'obstacle_free_width_float':
            if edge_attrs['obstacle_free_width_float'] < user_model.minimum_width:
                if result_df.at[row_id, 'include'] == 0:
                    if 'curb_height_max' in edge_attrs.keys():
                        if edge_attrs['curb_height_max'] is not None:
                            if edge_attrs['curb_height_max'] <= user_model.maximum_height:
                                result_df.at[row_id, 'include'] = 1
                        else:
                            result_df.at[row_id, 'include'] = 1
                    else:
                        result_df.at[row_id, 'include'] = 1
                result_df.at[row_id, 'obstacle_free_width_float'] = 2.0
                result.append(('add_width', (edge_attrs['index_position'], row.geometry),
                               2.0 - edge_attrs['obstacle_free_width_float'], 'success'))
            else:
                if result_df.at[row_id, 'include'] == 1:
                    result_df.at[row_id, 'include'] = 0
                result_df.at[row_id, 'obstacle_free_width_float'] = 0.6
                result.append(('sub_width', (edge_attrs['index_position'], row.geometry),
                               edge_attrs['obstacle_free_width_float'] - 0.6, 'success'))
        else:
            assert False
    return result_df, result


def to_dataframe(dataframe: GeoDataFrame, graph: TypedMultiGraph, route: Iterable[Edge]):
    indices = [graph.get_edge_data(edge)['index_position'] for edge in route]
    return dataframe.iloc[indices]


def compare_objectives(obj1: Objective, obj2: Objective, eps: float = 1e-6) -> int:
    """
    Compare two objectives (violation, num_changes, route_diff) lexicographically,
    treating the float components as equal if they differ by less than eps.
    """
    for idx, (a, b) in enumerate(zip(obj1, obj2)):
        if idx == 1:
            # exact comparison for the integer component
            if a < b:
                return -1
            if a > b:
                return 1
        else:
            # approximate comparison for floats
            diff = a - b
            if abs(diff) < eps:
                continue
            return -1 if diff < 0 else 1
    return 0


def compute_objective(instance: Instance, encoding: Iterable[Modification], fact_route: Iterable[Edge]):
    if fact_route is None:
        return 1e6, 1e6, 1e6
    difference = route_difference(instance.foil_route, fact_route, instance.graph)
    return max(difference - instance.delta, 0), len(encoding), difference


def get_weight(graph: TypedMultiGraph, encoding: Iterable[Modification], user_model: UserModel,
               path: Iterable[Edge]) -> float:
    modified_graph = _adjust_weight(graph, encoding, user_model)
    modified_graph = _remove_edges(modified_graph)
    return sum(modified_graph.get_edge_data(edge)['my_weight'] for edge in path)


def find_path(initial_G: TypedMultiGraph, pertubations: Iterable[Modification],
              user_model: UserModel, origin_node: NodeName, dest_node: NodeName):
    modified_G = _adjust_weight(initial_G, pertubations, user_model)
    modified_G = _remove_edges(modified_G)
    router = Router()
    route = router.get_route(modified_G, origin_node, dest_node)
    return route


def find_path_fast(graph: TypedMultiGraph, user_model: UserModel,
                   origin_node: NodeName, dest_node: NodeName, encoding: Iterable[Modification],
                   directional: bool = False):
    _modify_graph(graph, user_model, encoding, directional)
    removed = _remove_edges_saving(graph)
    router = Router()
    route = router.get_route(graph, origin_node, dest_node)
    _restore_edges(graph, removed)
    _restore_graph(graph, user_model, encoding, directional)
    return route


def find_path_fast2(graph: TypedMultiGraph, user_model: UserModel,
                    origin_node: NodeName, dest_node: NodeName, encoding: Iterable[Modification]):
    _modify_graph2(graph, user_model, encoding)
    router = Router()
    route = router.get_route(graph, origin_node, dest_node)
    _restore_graph2(graph, user_model, encoding)
    return route


def find_path_and_length_fast(graph: TypedMultiGraph, user_model: UserModel, foil_path: Iterable[Edge],
                              encoding: Iterable[Modification], origin_node: NodeName,
                              dest_node: NodeName, directional: bool = False):
    _modify_graph(graph, user_model, encoding, directional)
    removed = _remove_edges_saving(graph)
    router = Router()
    route = router.get_route(graph, origin_node, dest_node)
    foil_len = _route_length(graph, foil_path)
    fact_len = _route_length(graph, route)
    _restore_edges(graph, removed)
    _restore_graph(graph, user_model, encoding, directional)
    return route, foil_len, fact_len


def find_path_and_length_fast2(graph: TypedMultiGraph, user_model: UserModel, foil_path: Iterable[Edge],
                               encoding: Iterable[Modification], origin_node: NodeName,
                               dest_node: NodeName):
    _modify_graph2(graph, user_model, encoding)
    router = Router()
    route = router.get_route(graph, origin_node, dest_node)
    foil_len = _route_length(graph, foil_path)
    fact_len = _route_length(graph, route)
    _restore_graph2(graph, user_model, encoding)
    return route, foil_len, fact_len


@overload
def _route_length(graph: TypedMultiGraph, route: Iterable[Edge]) -> float: ...


@overload
def _route_length(graph: TypedMultiGraph, route: None) -> None: ...


def _route_length(graph: TypedMultiGraph, route: Iterable[Edge] | None) -> float | None:
    return sum(_get_weight_fallback(graph, edge) for edge in route) if route is not None else None


def _get_weight_fallback(graph: TypedMultiGraph, edge: Edge) -> float:
    try:
        return graph.get_edge_data(edge)['my_weight']
    except KeyError:
        return 1_000


def route_difference(route_1: Iterable[Edge], route_2: Iterable[Edge], graph: TypedMultiGraph,
                     len_col: str = 'mm_len') -> float:
    common_l = 0.0
    l1 = sum(graph.get_edge_data(edge)[len_col] for edge in route_1)
    l2 = sum(graph.get_edge_data(edge)[len_col] for edge in route_2)
    for edge in route_1:
        if edge in route_2:
            common_l += graph.get_edge_data(edge)[len_col]
    return 1 - 2 * common_l / (l1 + l2)


def _adjust_weight(G: TypedMultiGraph, pertubations: Iterable[Modification], user_model: UserModel):
    modified_G = G.copy()
    for var in pertubations:
        edge = var[0]
        edge_attrs = modified_G.get_edge_data(edge)
        if var[1] == 'path_type':
            if user_model.path_preference == edge_attrs['path_type']:
                edge_attrs['my_weight'] /= user_model.preference_weight
            else:
                edge_attrs['my_weight'] *= user_model.preference_weight
        elif var[1] == 'curb_height_max':
            if edge_attrs['curb_height_max'] > user_model.maximum_height:
                edge_attrs['include'] = 1
            else:
                edge_attrs['include'] = 0
        elif var[1] == 'obstacle_free_width_float':
            if edge_attrs['obstacle_free_width_float'] < user_model.minimum_width:
                edge_attrs['include'] = 1
            else:
                edge_attrs['include'] = 0
    return modified_G


def _remove_edges(G: TypedMultiGraph):
    for edge in list(G.edges):
        if G.get_edge_data(edge)['include'] == 0:
            G.remove_edge(edge)
    return G


def _get_edges_to_modify(
        graph: TypedMultiGraph, edge: Edge, directional: bool) -> list[Edge]:
    edges_to_modify = [edge]
    if not directional:
        u, v, _ = edge
        if graph.inner.has_edge(v, u):
            for rev_idx in graph.inner[v][u]:
                edges_to_modify.append((v, u, rev_idx))
    return edges_to_modify


def _modify_graph(graph: TypedMultiGraph, user_model: UserModel, encoding: Iterable[Modification], directional: bool):
    for edge, attribute in encoding:
        edges_to_modify = _get_edges_to_modify(graph, edge, directional)

        for e in edges_to_modify:
            edge_attrs = graph.get_edge_data(e)
            if attribute == 'path_type':
                if user_model.path_preference == edge_attrs['path_type']:
                    edge_attrs['my_weight'] /= user_model.preference_weight
                else:
                    edge_attrs['my_weight'] *= user_model.preference_weight
            elif attribute == 'curb_height_max':
                if edge_attrs['curb_height_max'] > user_model.maximum_height:
                    edge_attrs['include'] = 1
                else:
                    edge_attrs['include'] = 0
            elif attribute == 'obstacle_free_width_float':
                if edge_attrs['obstacle_free_width_float'] < user_model.minimum_width:
                    edge_attrs['include'] = 1
                else:
                    edge_attrs['include'] = 0
            else:
                assert False


def _modify_graph2(graph: TypedMultiGraph, user_model: UserModel, encoding: Iterable[Modification]):
    for edge, attribute in encoding:
        edge_attrs = graph.get_edge_data(edge)
        if attribute == 'path_type':
            if user_model.path_preference == edge_attrs['path_type']:
                edge_attrs['my_weight'] /= user_model.preference_weight
            else:
                edge_attrs['my_weight'] *= user_model.preference_weight
        elif attribute == 'curb_height_max':
            if edge_attrs['curb_height_max'] > user_model.maximum_height:
                edge_attrs['my_weight'] /= utility.ABSENT_COEFFICIENT
            else:
                edge_attrs['my_weight'] *= utility.ABSENT_COEFFICIENT
        elif attribute == 'obstacle_free_width_float':
            if edge_attrs['obstacle_free_width_float'] < user_model.minimum_width:
                edge_attrs['my_weight'] /= utility.ABSENT_COEFFICIENT
            else:
                edge_attrs['my_weight'] *= utility.ABSENT_COEFFICIENT
        else:
            assert False


def _restore_graph2(graph: TypedMultiGraph, user_model: UserModel, encoding: Iterable[Modification]):
    for edge, attribute in encoding:
        edge_attrs = graph.get_edge_data(edge)
        if attribute == 'path_type':
            if user_model.path_preference == edge_attrs['path_type']:
                edge_attrs['my_weight'] *= user_model.preference_weight
            else:
                edge_attrs['my_weight'] /= user_model.preference_weight
        elif attribute == 'curb_height_max':
            if edge_attrs['curb_height_max'] > user_model.maximum_height:
                edge_attrs['my_weight'] *= utility.ABSENT_COEFFICIENT
            else:
                edge_attrs['my_weight'] /= utility.ABSENT_COEFFICIENT
        elif attribute == 'obstacle_free_width_float':
            if edge_attrs['obstacle_free_width_float'] < user_model.minimum_width:
                edge_attrs['my_weight'] *= utility.ABSENT_COEFFICIENT
            else:
                edge_attrs['my_weight'] /= utility.ABSENT_COEFFICIENT
        else:
            assert False


def _restore_graph(graph: TypedMultiGraph, user_model: UserModel, encoding: Iterable[Modification], directional: bool):
    for edge, attribute in encoding:
        edges_to_restore = _get_edges_to_modify(graph, edge, directional)

        for e in edges_to_restore:
            edge_attrs = graph.get_edge_data(e)
            if attribute == 'path_type':
                if user_model.path_preference == edge_attrs['path_type']:
                    edge_attrs['my_weight'] *= user_model.preference_weight
                else:
                    edge_attrs['my_weight'] /= user_model.preference_weight
            elif attribute == 'curb_height_max':
                if edge_attrs['curb_height_max'] > user_model.maximum_height:
                    edge_attrs['include'] = 0
                else:
                    edge_attrs['include'] = 1
            elif attribute == 'obstacle_free_width_float':
                if edge_attrs['obstacle_free_width_float'] < user_model.minimum_width:
                    edge_attrs['include'] = 0
                else:
                    edge_attrs['include'] = 1
            else:
                assert False


def _remove_edges_saving(graph: TypedMultiGraph):
    removed: list[tuple[Edge, Mapping[str, Any]]] = []
    for edge in list(graph.edges):
        data = graph.get_edge_data(edge)
        if data['include'] == 0:
            graph.remove_edge(edge)
            removed.append((edge, data))
    return removed


def _restore_edges(graph: TypedMultiGraph, edges: Iterable[tuple[Edge, Mapping[str, Any]]]):
    for edge, attributes in edges:
        graph.add_edge(edge, attributes)
