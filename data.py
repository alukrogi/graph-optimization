# Standard library and path imports
import json
import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, no_type_check

import geopandas as gpd
import momepy  # type: ignore
import networkx as nx
import numpy as np
import pandas as pd
# Third-party library imports
from networkx import MultiGraph
from shapely import wkt

import utility
from graph_types import NodeName, TypedMultiGraph
# Local or project-specific imports
from router import Router
from utility import FullInstanceData, Instance
from validation.dataparser import handle_weight


@no_type_check
def create_network_graph2(df):
    df = df.copy()
    df['oneway'] = np.where(df['bikepath_id'].isna(), False, True)
    return momepy.gdf_to_nx(
        df,
        approach="primal",
        multigraph=True,
        directed=True,
        oneway_column="oneway", preserve_index=True)


@no_type_check
def read_data(args):
    basic_network_path = args['basic_network_path']
    foil_json_path = args['foil_json_path']
    df_path_foil_path = args['df_path_foil_path']
    gdf_coords_path = args['gdf_coords_path']
    df = gpd.read_file(basic_network_path)
    with open(foil_json_path, 'r') as f:
        path_foil = json.load(f)
    df_path_foil = gpd.read_file(df_path_foil_path)
    gdf_coords_loaded = pd.read_csv(gdf_coords_path, sep=';')
    gdf_coords_loaded['geometry'] = gdf_coords_loaded['geometry'].apply(wkt.loads)
    gdf_coords_loaded = gpd.GeoDataFrame(gdf_coords_loaded, geometry='geometry')
    return df, path_foil, df_path_foil, gdf_coords_loaded


def read_instance(cli_args):
    basic_network_path = cli_args.basic_network_path
    foil_json_path = cli_args.foil_json_path
    df_path_foil_path = cli_args.df_path_foil_path
    gdf_coords_path = cli_args.gdf_coords_path
    meta_data_path = cli_args.meta_data_path
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    # Profile settings
    user_model = meta_data["user_model"]
    meta_map = meta_data["map"]
    attrs_variable_names = user_model["attrs_variable_names"]
    route_error_delta = user_model["route_error_threshold"]
    # Demo route
    args = {
        'basic_network_path': basic_network_path,
        'foil_json_path': foil_json_path,
        'df_path_foil_path': df_path_foil_path,
        'gdf_coords_path': gdf_coords_path,
        'heuristic': 'dijkstra',
        'heuristic_f': 'my_weight',
        'jobs': 10,
        'attrs_variable_names': attrs_variable_names,
        "user_model": user_model,
        "meta_map": meta_map
    }

    router = Router()

    df, path_foil, df_path_foil, gdf_coords_loaded = read_data(args)
    df_copy = deepcopy(df)
    df_copy = handle_weight(df_copy, user_model)
    G = create_network_graph2(df_copy)
    assert isinstance(G, MultiGraph)
    origin_node, dest_node, origin_node_loc, dest_node_loc, gdf_coords = router.set_o_d_coords(G, gdf_coords_loaded)

    foil_edges = utility.node_path_to_edges(tuple(map(tuple, path_foil)))
    user_model_class = utility.UserModel(
        user_model['min_sidewalk_width'],
        user_model['max_curb_height'],
        user_model['walk_bike_preference'],
        user_model['walk_bike_preference_weight_factor'])
    graph = TypedMultiGraph(G)
    remove_loops(graph)
    remove_nans(graph)
    graph = drop_unreachable(graph, foil_edges[0][0])
    remove_leaves(graph, frozenset(map(tuple, path_foil)))
    set_include(graph, user_model_class, user_model)
    remove_useless_attributes(graph)
    return FullInstanceData(Instance(graph, foil_edges, user_model_class, route_error_delta),
                            gdf_coords, origin_node_loc, dest_node_loc, meta_data, df_copy, df_path_foil)


@no_type_check
def read_instance2(n: int, m: int):
    base = f"data/train/routes/osdpm_{n}_{m}"
    foil_json_path = f"{base}/foil_route.json"
    df_path_foil_path = f"{base}/foil_route.gpkg"
    gdf_coords_path = f"{base}/route_start_end.csv"
    meta_data_path = f"{base}/metadata.json"
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
    meta_map = meta_data["map"]
    basic_network_path = f"data/train/maps/{meta_map['map_name']}"

    cli_args = SimpleNamespace(
        basic_network_path=basic_network_path,
        foil_json_path=foil_json_path,
        df_path_foil_path=df_path_foil_path,
        gdf_coords_path=gdf_coords_path,
        meta_data_path=meta_data_path,
    )

    return read_instance(cli_args)


def read_instance3(folder, route_name, route_id):
    basic_network_path = f'{folder}/network_{route_name}_{route_id}.gpkg'
    foil_json_path = f'{folder}/route_nodes_{route_name}_{route_id}.json'
    df_path_foil_path = f'{folder}/route_{route_name}_{route_id}.gpkg'
    gdf_coords_path = f'{folder}/route_{route_name}_{route_id}_start_end.csv'
    meta_data_path = 'data/metadata_demo_walk_0.json'
    cli_args = SimpleNamespace(
        basic_network_path=basic_network_path,
        foil_json_path=foil_json_path,
        df_path_foil_path=df_path_foil_path,
        gdf_coords_path=gdf_coords_path,
        meta_data_path=meta_data_path,
    )

    return read_instance(cli_args)


def drop_unreachable(graph: TypedMultiGraph, node: NodeName):
    inner = graph.inner
    for component in nx.weakly_connected_components(inner):  # type: ignore
        if node in component:
            return TypedMultiGraph(inner.subgraph(component).copy())  # type: ignore
    assert False


def remove_loops(graph: TypedMultiGraph):
    while True:
        for edge in list(graph.edges):
            if edge[0] == edge[1]:
                graph.remove_edge(edge)
                break
        else:
            break


def remove_nans(graph: TypedMultiGraph):
    for edge in list(graph.edges):
        data = graph.get_edge_data(edge)
        if math.isnan(data['length']):
            graph.remove_edge(edge)
        if math.isnan(data['curb_height_max']):
            del data['curb_height_max']


def set_include(graph: TypedMultiGraph, user_model: utility.UserModel, model: Mapping[str, Any]):
    for edge in graph.edges:
        include = True
        data = graph.get_edge_data(edge)
        if data.get('curb_height_max', 0) > user_model.maximum_height:
            include = False
        if data['obstacle_free_width_float'] < user_model.minimum_width:
            include = False
        data['include'] = 1 if include else 0
        weight = data['length']
        if data['crossing'] == 'Yes':
            weight *= model['crossing_weight_factor']
        if data['path_type'] == model['walk_bike_preference']:
            weight *= model['walk_bike_preference_weight_factor']
        data['my_weight'] = weight


def set_include2(graph: TypedMultiGraph, user_model: utility.UserModel, model: Mapping[str, Any]):
    for edge in graph.edges:
        include = 0
        data = graph.get_edge_data(edge)
        if data.get('curb_height_max', 0) > user_model.maximum_height:
            include += 1
        if data['obstacle_free_width_float'] < user_model.minimum_width:
            include += 1
        data['include'] = 1 if include else 0
        weight = data['length']
        if data['crossing'] == 'Yes':
            weight *= model['crossing_weight_factor']
        if data['path_type'] == model['walk_bike_preference']:
            weight *= model['walk_bike_preference_weight_factor']
        data['my_weight'] = weight * utility.ABSENT_COEFFICIENT ** include


def remove_leaves(graph: TypedMultiGraph, foil_path: Iterable[NodeName]):
    while True:
        removed = False
        for node in list(graph.nodes):
            if graph.get_degree(node) <= 1 and node not in foil_path:
                graph.remove_node(node)
                removed = True
        if not removed:
            break


def remove_useless_attributes(graph: TypedMultiGraph):
    used = frozenset(('path_type', 'curb_height_max', 'obstacle_free_width_float', 'include', 'length', 'my_weight',
                      'index_position', 'mm_len'))
    for edge in graph.edges:
        data = graph.get_edge_data(edge)
        for key in list(data.keys()):
            if key not in used:
                del data[key]
