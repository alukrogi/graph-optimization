import argparse
import json
import os
from argparse import Namespace
from typing import Any

from geopandas import GeoDataFrame
from shapely import to_wkt

import data
from objective import create_output
from run_lns import run_multistart_LNS
from utility import calculate_segment_time_limit
from validation.dataparser import convert


def get_results(cli_args: Namespace):
    full_instance_data = data.read_instance(cli_args)
    total_time_limit = cli_args.time_limit
    segment_time_limit = calculate_segment_time_limit(total_time_limit)
    solution, time_to_best = run_multistart_LNS(full_instance_data, segment_time_limit)

    op_list, map_df = create_output(
        full_instance_data.df,
        full_instance_data.instance.graph,
        full_instance_data.instance.user_model,
        solution.encoding
    )

    available_op = [
        (op[0], (convert(op[1][0]), to_wkt(op[1][1], rounding_precision=-1, trim=False)), convert(op[2]), op[3]) for op
        in op_list if op[3] == "success"]

    return map_df, available_op


def store_results(output_path: str, map_df: GeoDataFrame, op_list: list[tuple[str, tuple[int, Any], float | str, str]]):
    map_df_path = os.path.join(output_path, "map_df.gpkg")
    op_list_path = os.path.join(output_path, "op_list.json")

    map_df.to_file(map_df_path, driver='GPKG')
    with open(op_list_path, 'w') as f:
        json.dump(op_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_data_path", type=str, required=True)
    parser.add_argument("--basic_network_path", type=str, required=True)
    parser.add_argument("--foil_json_path", type=str, required=True)
    parser.add_argument("--df_path_foil_path", type=str, required=True)
    parser.add_argument("--gdf_coords_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--time_limit", type=str, default=300.0)
    args = parser.parse_args()

    map_df, op_list = get_results(args)
    store_results(args.output_path, map_df, op_list)
