import json
import warnings
from copy import deepcopy

import geopandas as gpd
import pandas as pd
from shapely import wkt

from objective import create_output, compare_objectives
from utility import Solution, FullInstanceData, Objective
from validation.dataparser import create_network_graph, handle_weight_with_recovery, load_op_list
from validation.graph_op import graphOperator, pertub_with_op_list
from validation.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
from validation.router import Router

# suppress _only_ the “convert_dtype parameter is deprecated” FutureWarning from GeoPandas
warnings.filterwarnings(
    "ignore",
    message=r".*convert_dtype parameter is deprecated.*",
    category=FutureWarning,
    module=r"geopandas\.geoseries"
)


def read_data(basic_network_path: str, foil_json_path: str, df_path_foil_path: str, gdf_coords_path: str):
    df = gpd.read_file(basic_network_path)
    with open(foil_json_path, 'r') as f:
        path_foil = json.load(f)

    df_path_foil = gpd.read_file(df_path_foil_path)
    gdf_coords_loaded = pd.read_csv(gdf_coords_path, sep=';')

    gdf_coords_loaded['geometry'] = gdf_coords_loaded['geometry'].apply(wkt.loads)
    gdf_coords_loaded = gpd.GeoDataFrame(gdf_coords_loaded, geometry='geometry')

    return df, path_foil, df_path_foil, gdf_coords_loaded


def validate_solution(solution: Solution, full_instance_data: FullInstanceData):
    instance = full_instance_data.instance
    df_copy = deepcopy(full_instance_data.df)
    df_path_foil = full_instance_data.df_path_foil
    meta_data = full_instance_data.meta_data
    user_model = meta_data["user_model"]
    meta_map = meta_data["map"]
    attrs_variable_names = user_model["attrs_variable_names"]
    route_error_delta = user_model["route_error_threshold"]

    op_list, _ = create_output(full_instance_data.df, instance.graph, instance.user_model, solution.encoding)

    operator = graphOperator()
    df_p, op = pertub_with_op_list(operator, op_list, df_copy)
    df_p = handle_weight_with_recovery(df_p, user_model)

    _, G = create_network_graph(df_p)
    router_h = Router(heuristic='dijkstra', CRS=meta_map["CRS"], CRS_map=meta_map["CRS_map"])
    origin_node, dest_node, origin_node_loc, dest_node_loc, gdf_coords = router_h.set_o_d_coords(
        G, full_instance_data.gdf_coords)

    path_fact, G_path_fact, df_path_fact = router_h.get_route(
        G, origin_node, dest_node, 'my_weight')
    difference = 1 - common_edges_similarity_route_df_weighted(df_path_fact, df_path_foil, attrs_variable_names)
    calculated_obj = max(difference - route_error_delta, 0), len(op_list), difference
    is_obj_correct = compare_objectives(calculated_obj, solution.objective) == 0
    return is_obj_correct, calculated_obj, G_path_fact


def validate(
        basic_network_path: str,
        foil_json_path: str,
        df_path_foil_path: str,
        gdf_coords_path: str,
        meta_data_path: str,
        df_perturbed_path: str,
        op_list_path: str,
        objective: Objective
):
    meta_data = json.load(open(meta_data_path))
    user_model = meta_data["user_model"]
    meta_map = meta_data["map"]

    attrs_variable_names = user_model["attrs_variable_names"]
    route_error_delta = user_model["route_error_threshold"]

    df, path_foil, df_path_foil, gdf_coords_loaded = read_data(
        basic_network_path, foil_json_path, df_path_foil_path, gdf_coords_path
    )

    df_copy = deepcopy(df)
    operator = graphOperator()
    op_list = load_op_list(op_list_path)
    df_p, _ = pertub_with_op_list(operator, op_list, df_copy)
    df_p = handle_weight_with_recovery(df_p, user_model)

    router_h = Router(
        heuristic="dijkstra",
        CRS=meta_map["CRS"],
        CRS_map=meta_map["CRS_map"],
    )

    def route_diff(df_variant):
        _, G = create_network_graph(df_variant)
        origin_node, dest_node, *_ = router_h.set_o_d_coords(G, gdf_coords_loaded)
        _, _, df_path_fact = router_h.get_route(G, origin_node, dest_node, "my_weight")
        return 1 - common_edges_similarity_route_df_weighted(
            df_path_fact, df_path_foil, attrs_variable_names
        )

    difference = route_diff(df_p)
    calculated_obj1 = (max(difference - route_error_delta, 0), len(op_list), difference)

    df_p2 = gpd.read_file(df_perturbed_path)
    sub_op_list = get_virtual_op_list(df, df_p2, attrs_variable_names)
    graph_error = sum(1 for op in sub_op_list if op[3] == "success")

    difference = route_diff(df_p2)
    calculated_obj2 = (max(difference - route_error_delta, 0), graph_error, difference)
    return compare_objectives(calculated_obj1, calculated_obj2) == 0 and compare_objectives(calculated_obj1,
                                                                                            objective) == 0, calculated_obj1
