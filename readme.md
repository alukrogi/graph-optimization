# HGLM – Submission

A Python 3.12 implementation of multi‑start Large‑Neighborhood Search (LNS) to produce counterfactual maps where a foil
route becomes (near-)optimal under a user model.

## Dependencies

- geopandas==0.13.2
- fiona==1.9.6
- momepy>=0.8.1
- networkx==3.1
- numpy==1.26.4
- pandas==2.0.3
- shapely==2.0.6

Install with:

```bash
pip install -r requirements.txt
````

## Usage

### Single instance

```bash
python main.py \
  --meta_data_path <metadata.json> \
  --basic_network_path <map_folder> \
  --foil_json_path <foil_route.json> \
  --df_path_foil_path <foil_route.gpkg> \
  --gdf_coords_path <route_start_end.csv> \
  --output_path out \
  --time_limit 300.0
```

Outputs `out/map_df.gpkg` and `out/op_list.json`.

### Batch validation

```bash
python run_and_validate_osdpm.py
```

Generates `out/validation_summary.csv` for all training instances.

### Core LNS runner

```bash
python run_lns.py
```

Runs the core multi‑start LNS algorithm on sample instances; prints the best objective, wall‑clock duration, and time‑to‑best for each run.

### `out/` directory

The `out/` folder contains all results produced by running `run_and_validate_osdpm.py` on a laptop with an Intel i5‑13500H CPU and 16 GB of RAM. 
The time limit for each run was 100 seconds.
It also includes a brief description of proved lower bounds for the training instances.

## Structure

```
.
├── main.py
├── run_and_validate_osdpm.py
├── run_lns.py
├── ...
├── out/
├── data/
└── validation/
```


