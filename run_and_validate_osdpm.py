import json
import os
import time
from types import SimpleNamespace

import pandas as pd
from shapely import to_wkt

import data
from objective import create_output
from run_lns import run_multistart_LNS
from utility import calculate_segment_time_limit
from validation.dataparser import convert
from validation.validator import validate


def get_results(cli_args: SimpleNamespace):
    t0 = time.monotonic()
    full_instance_data = data.read_instance(cli_args)
    read_data_time = time.monotonic() - t0
    total_time_limit = cli_args.time_limit - read_data_time
    segment_time_limit = calculate_segment_time_limit(total_time_limit)  # three segments + space for overhead
    solution, time_to_best = run_multistart_LNS(full_instance_data, segment_time_limit, cli_args.n_workers)

    map_df, op_list = create_output(
        full_instance_data.df,
        full_instance_data.instance.graph,
        full_instance_data.meta_data["user_model"],
        solution.encoding
    )

    available_op = [
        (op[0], (convert(op[1][0]), to_wkt(op[1][1], rounding_precision=-1, trim=False)), convert(op[2]), op[3]) for op
        in op_list if op[3] == "success"]

    return map_df, available_op, solution.objective, time_to_best


def save_validation_summary(
        results: dict[str, tuple[tuple[bool, tuple[float, ...]], float, float]],
        csv_path: str,
        unpack_validation: bool = True
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        results,
        orient='index',
        columns=['validation', 'elapsed, s', 'time_to_best, s']
    ).reset_index().rename(columns={'index': 'instance'})

    if unpack_validation:
        # Split validation -> success + info tuple
        df[['success', 'info']] = pd.DataFrame(
            df['validation'].tolist(), index=df.index
        )
        # Split info tuple into its components with meaningful names
        info_cols = ['penalty', 'graph_error', 'route_difference']
        df[info_cols] = pd.DataFrame(
            df['info'].tolist(), index=df.index
        )
        df = df.drop(columns=['validation', 'info'])
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    results = {}
    for n in range(0, 5):
        for m in range(1, 6):
            start = time.monotonic()
            inst_name = f"osdpm_{n}_{m}"
            base = f"data/train/routes/{inst_name}"
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
                output_path="out",
                time_limit=300.0,
                n_workers=max(1, (os.cpu_count() or 1) - 1)
            )
            print(f"Running instance {inst_name} ...")
            result_map_df, result_op_list, objective, time_to_best = get_results(cli_args)
            map_df_path = os.path.join(cli_args.output_path, f"map_df_{n}_{m}.gpkg")
            op_list_path = os.path.join(cli_args.output_path, f"op_list_{n}_{m}.json")

            result_map_df.to_file(map_df_path, driver='GPKG')
            with open(op_list_path, 'w') as f:
                json.dump(result_op_list, f)
            elapsed = time.monotonic() - start
            print(f"Instance {inst_name} result: {objective} | {elapsed} s | {time_to_best} s")
            results[f"{inst_name}"] = validate(basic_network_path, foil_json_path, df_path_foil_path, gdf_coords_path,
                                               meta_data_path, map_df_path, op_list_path,
                                               objective), elapsed, time_to_best
    save_validation_summary(results, "out/validation_summary.csv")
