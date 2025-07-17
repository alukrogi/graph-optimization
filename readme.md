# HGLM – Submission

A Python 3.12 implementation of multi‑start Large Neighborhood Search (LNS) to produce counterfactual maps where a foil
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

Run the algorithm on a single instance using the following command:

```bash
python main.py \
  --meta_data_path <metadata.json> \
  --basic_network_path <map_folder> \
  --foil_json_path <foil_route.json> \
  --df_path_foil_path <foil_route.gpkg> \
  --gdf_coords_path <route_start_end.csv> \
  --output_path <out_dir> \
  --time_limit 300.0
```

**Example:**

```bash
python main.py --meta_data_path data/train/routes/osdpm_1_4/metadata.json --basic_network_path data/train/maps/osdpm_segment_1.gpkg --foil_json_path data/train/routes/osdpm_1_4/foil_route.json --df_path_foil_path data/train/routes/osdpm_1_4/foil_route.gpkg --gdf_coords_path data/train/routes/osdpm_1_4/route_start_end.csv --output_path out --time_limit 100.0
```

This generates `out/map_df.gpkg` and `out/op_list.json` for the `osdpm_1_4` instance, computed within the 100-second
time limit.

### Batch validation

```bash
python run_and_validate_osdpm.py
```

Generates `out/validation_summary.csv` for all training instances.

### Core LNS runner

```bash
python run_lns.py
```

Runs the core multi‑start LNS algorithm on sample instances; prints the best objective, wall‑clock duration, and
time‑to‑best for each run.

### `out/` directory

The `out/` folder contains all results produced by running `run_and_validate_osdpm.py` on a laptop with an Intel
i5‑13500H CPU and 16 GB of RAM.
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

## Notes

- **Subgraph selection in `create_network_graph(df)`**
  ```python
  S_sel = [G_sel.subgraph(c).copy()
           for c in sorted(nx.connected_components(G_sel),
                           key=len, reverse=True)]
  # Networks may fragment into multiple disconnected subgraphs.
  # If `my_areas` is set, only the two largest subgraphs are kept.
  # Demo data is fully connected, so this step is skipped:
  G_con = G
  if len(S_sel) > 1:
      G_sel_con = nx.compose(S_sel[0], S_sel[1])
  else:
      G_sel_con = G_sel
    ```

_**Caution:** The validation pipeline applies subgraph‑selection (keeping only the largest connected components),
whereas our optimization routines operate on the **entire** graph. If a data contains multiple disconnected components,
this mismatch may cause discrepancies in validation results._

- **Graph directionality mismatch in modifications:** All our experiments have been conducted in “directional”
  mode—modifications affect only the specified (u→v) edges—whereas the validation pipeline and output operator list
  assume an undirected graph, mirroring each change onto both directions of any bi‑directional edge, potentially causing
  discrepancies. We’ve now added a `directional: bool` flag in `objective.py` (default`False` corresponding to the
  submission template) to enable “undirectional” optimization (i.e. automatic dual‑direction updates) when set to
  `False`.

- **Time-limit sensitivity:** The current implementation performs its multi-start, repair, and destroy iterations
  without internal early exits, which means it may struggle—or even fail to produce a valid counterfactual—under very
  tight time budgets (e.g., under a few seconds). However, it generally should run reliably with the 300s limit used in
  the competition.

- **Unused code parts:** Some modules and functions remain in the codebase but aren’t invoked in the current LNS
  flow—they were used in initial experiments but were not integrated into the final scheme. They’re retained for
  potential inclusion in future hybrid approaches.

- **`mip.py` reference:** The `mip.py` file contains a MIP formulation of the “directional” problem. It isn’t invoked by
  the LNS runner and is provided mainly as an optional reference for alternative methods.

- **Validation toggle (`apply_validate`):** By default, every solution is passed through `validate_solution` to catch
  inconsistencies. If you need to skip this step, open **LNS.py** and, in the `update_best(...)` function, set
  `apply_validate=False`. There’s no config option—validation is assumed mandatory, so disable it in the source only if
  validation step doesn’t match your intended evaluation procedure.