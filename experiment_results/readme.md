# Experiment Results

This directory contains the complete results of our computational study on the IJCAI‑25 Counterfactual Routing Problem (CRP).
We present a detailed comparison of two heuristic algorithms — **Tree Search (TS)** and **Destroy‑and‑Repair (DR)**
— against the best‑known solutions (BKS) and optimality bounds obtained via our MIP model.

---

## Contents

* **results.csv**
  A CSV file with one row per test instance, including:

    * **Instance**: unique identifier for each test case
    * **BKS**: best‑known solution value (number of modifications, penalties $\Delta=0$)
    * **OPT Proven**: whether we proved that the BKS is optimal using our MIP model
    * **TreeSearch**: solution value returned by the deterministic Tree Search heuristic
    * **DR\_Min**: best (minimum) solution value over 10 independent runs of multi-start Destroy‑and‑Repair heuristic
    * **DR\_Avg**: average solution value over those 10 runs
    * **DR\_Max**: worst (maximum) solution value over those 10 runs

---

## Instance Generation and Experimental Setup

All experiments were run on a desktop with an Intel Core i7‑13700 CPU, 32 GB of RAM, using 23 threads for multi-start
DR.

We evaluated on:

* **demo** (3 provided demo instances)
* **osdpm** (25 Amsterdam Osdorp‑Midden instances)
* **bbox1–bbox3\_0.15\_bike** (six sets generated
  via [Accessible Route Planning](https://github.com/Amsterdam-AI-Team/Accessible_Route_Planning): 10 instances each,
  except 9 for bbox1)

The **bbox** sets cover three different-size areas of Amsterdam. For all sets except **bbox1**, foil routes correspond
to the shortest feasible paths in a perturbed graph, making these instances more challenging. We applied the same user
metadata (user model) as in the demo instances; for **bbox3\_0.15\_bike**, the $\delta$ is set to 0.15 and the
user path preference to “bike.”

## Aggregated Performance Overview

The table below summarizes each instance set’s overall performance. Gaps for TS and DR are computed as

$$
\text{Gap}(\%) = \frac{\text{HeuristicValue} - \text{BKS}}{\text{BKS}} \times 100\%  
$$

| Instance Set      | % OPT Proven | Avg. Gap TS | Avg. Gap DR |
|:------------------|:------------:|:-----------:|:-----------:|
| demo              |    100 %     |     0 %     |     0 %     |
| osdpm             |     92 %     |     2 %     |     0 %     |
| bbox1             |     67 %     |     6 %     |     0 %     |
| bbox1-p           |     40 %     |     8 %     |     0 %     |
| bbox2-short       |     90 %     |     0 %     |     0 %     |
| bbox2-long        |     20 %     |    48 %     |     0 %     |
| bbox3             |     0 %      |    111 %    |     2 %     |
| bbox3\_0.15\_bike |     0 %      |    155 %    |     3 %     |
| **TOTAL**         |   **54 %**   |  **38 %**   |  **0.5 %**  |

> **Notes:**
>
> * **% OPT Proven**: percentage of instances where our MIP model certified the BKS as optimal.
> * **Avg. Gap**: mean percentage deviation of the heuristic’s result from the BKS across all instances in the set.

---

## Detailed Results by Instance Sets

Below, instances are grouped by their test‑set prefixes. Cells in **bold** highlight cases where the optimality is
proven.

### Demo Instances

| Instance |   BKS | TS | DR Min | DR Avg | DR Max |
|:--------:|------:|---:|-------:|-------:|-------:|
| demo\_0  | **4** |  4 |      4 |      4 |      4 |
| demo\_1  | **3** |  3 |      3 |      3 |      3 |
| demo\_3  | **3** |  3 |      3 |      3 |      3 |

---

### OSDPM Sets

#### Set 0

|  Instance   |   BKS | TS | DR Min | DR Avg | DR Max |
|:-----------:|------:|---:|-------:|-------:|-------:|
| osdpm\_0\_1 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_0\_2 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_0\_3 |     3 |  3 |      3 |      3 |      3 |
| osdpm\_0\_4 | **2** |  2 |      2 |      2 |      2 |
| osdpm\_0\_5 | **2** |  2 |      2 |      2 |      2 |

#### Set 1

|  Instance   |   BKS | TS | DR Min | DR Avg | DR Max |
|:-----------:|------:|---:|-------:|-------:|-------:|
| osdpm\_1\_1 | **3** |  3 |      3 |      3 |      3 |
| osdpm\_1\_2 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_1\_3 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_1\_4 |     9 | 11 |      9 |      9 |      9 |
| osdpm\_1\_5 | **1** |  1 |      1 |      1 |      1 |

#### Set 2

|  Instance   |   BKS | TS | DR Min | DR Avg | DR Max |
|:-----------:|------:|---:|-------:|-------:|-------:|
| osdpm\_2\_1 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_2\_2 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_2\_3 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_2\_4 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_2\_5 | **3** |  3 |      3 |      3 |      3 |

#### Set 3

|  Instance   |   BKS | TS | DR Min | DR Avg | DR Max |
|:-----------:|------:|---:|-------:|-------:|-------:|
| osdpm\_3\_1 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_3\_2 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_3\_3 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_3\_4 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_3\_5 | **4** |  4 |      4 |      4 |      4 |

#### Set 4

|  Instance   |   BKS | TS | DR Min | DR Avg | DR Max |
|:-----------:|------:|---:|-------:|-------:|-------:|
| osdpm\_4\_1 | **4** |  5 |      4 |      4 |      4 |
| osdpm\_4\_2 | **4** |  4 |      4 |      4 |      4 |
| osdpm\_4\_3 | **3** |  3 |      3 |      3 |      3 |
| osdpm\_4\_4 | **1** |  1 |      1 |      1 |      1 |
| osdpm\_4\_5 | **1** |  1 |      1 |      1 |      1 |

---

### BBox1 Instances

| Instance  |   BKS | TS | DR Min | DR Avg | DR Max |
|-----------|------:|---:|-------:|-------:|-------:|
| bbox1\_1  | **6** |  6 |      6 |      6 |      6 |
| bbox1\_2  |    10 | 10 |     10 |     10 |     10 |
| bbox1\_3  | **2** |  2 |      2 |      2 |      2 |
| bbox1\_4  |     9 | 10 |      9 |      9 |      9 |
| bbox1\_5  | **7** |  7 |      7 |      7 |      7 |
| bbox1\_7  | **8** |  8 |      8 |      8 |      8 |
| bbox1\_8  | **4** |  4 |      4 |      4 |      4 |
| bbox1\_9  | **5** |  5 |      5 |      5 |      5 |
| bbox1\_10 |    10 | 14 |     10 |     10 |     10 |

### BBox1\_p Instances

| Instance   |   BKS | TS | DR Min | DR Avg | DR Max |
|------------|------:|---:|-------:|-------:|-------:|
| bbox1\_p1  |    17 | 17 |     17 |     17 |     17 |
| bbox1\_p2  |    11 | 13 |     11 |     11 |     11 |
| bbox1\_p3  | **2** |  2 |      2 |      2 |      2 |
| bbox1\_p4  |    18 | 19 |     18 |     18 |     18 |
| bbox1\_p5  | **7** |  7 |      7 |      7 |      7 |
| bbox1\_p6  | **9** | 10 |      9 |      9 |      9 |
| bbox1\_p7  |    10 | 11 |     10 |     10 |     10 |
| bbox1\_p8  |    14 | 18 |     14 |     14 |     14 |
| bbox1\_p9  | **7** |  7 |      7 |      7 |      7 |
| bbox1\_p10 |    10 | 11 |     10 |     10 |     10 |

---

### BBox2‑Short Instances

| Instance        |   BKS | TS | DR Min | DR Avg | DR Max |
|-----------------|------:|---:|-------:|-------:|-------:|
| bbox2-short\_1  | **3** |  3 |      3 |      3 |      3 |
| bbox2-short\_2  | **4** |  4 |      4 |      4 |      4 |
| bbox2-short\_3  |     7 |  7 |      7 |      7 |      7 |
| bbox2-short\_4  | **3** |  3 |      3 |      3 |      3 |
| bbox2-short\_5  | **6** |  6 |      6 |      6 |      6 |
| bbox2-short\_6  | **2** |  2 |      2 |      2 |      2 |
| bbox2-short\_7  | **3** |  3 |      3 |      3 |      3 |
| bbox2-short\_8  | **5** |  5 |      5 |      5 |      5 |
| bbox2-short\_9  | **9** |  9 |      9 |      9 |      9 |
| bbox2-short\_10 | **6** |  6 |      6 |      6 |      6 |

---

### BBox2‑Long Instances

| Instance       |    BKS | TS | DR Min | DR Avg | DR Max |
|----------------|-------:|---:|-------:|-------:|-------:|
| bbox2-long\_1  |     11 | 12 |     11 |     11 |     11 |
| bbox2-long\_2  |     16 | 18 |     16 |     16 |     16 |
| bbox2-long\_3  |      5 |  6 |      5 |      5 |      5 |
| bbox2-long\_4  |  **6** | 16 |      6 |      6 |      6 |
| bbox2-long\_5  |     14 | 20 |     14 |     14 |     14 |
| bbox2-long\_6  | **14** | 18 |     14 |     14 |     14 |
| bbox2-long\_7  |     14 | 34 |     14 |     14 |     14 |
| bbox2-long\_8  |     13 | 17 |     13 |     13 |     13 |
| bbox2-long\_9  |      7 |  9 |      7 |      7 |      7 |
| bbox2-long\_10 |      8 |  8 |      8 |      8 |      8 |

---

### BBox3 Instances

| Instance  | BKS |  TS | DR Min | DR Avg | DR Max |
|-----------|----:|----:|-------:|-------:|-------:|
| bbox3\_1  |  17 |  40 |     17 |   17.8 |     18 |
| bbox3\_2  |  33 |  69 |     33 |   33.8 |     35 |
| bbox3\_3  |  13 |  16 |     13 |     13 |     13 |
| bbox3\_4  |  34 | 118 |     34 |   34.7 |     37 |
| bbox3\_5  |  42 |  95 |     42 |   42.7 |     44 |
| bbox3\_6  |  41 |  79 |     41 |   42.3 |     43 |
| bbox3\_7  |  19 |  44 |     19 |     19 |     19 |
| bbox3\_8  |  19 |  36 |     19 |   19.5 |     20 |
| bbox3\_9  |  14 |  21 |     14 |     14 |     14 |
| bbox3\_10 |  25 |  52 |     25 |   25.4 |     26 |

---

### BBox3\_0.15\_bike Instances

| Instance              | BKS | TS | DR Min | DR Avg | DR Max |
|-----------------------|----:|---:|-------:|-------:|-------:|
| bbox3\_0.15\_bike\_1  |   4 |  5 |      4 |      4 |      4 |
| bbox3\_0.15\_bike\_2  |  16 | 37 |     16 |     16 |     16 |
| bbox3\_0.15\_bike\_3  |   5 | 11 |      5 |      5 |      5 |
| bbox3\_0.15\_bike\_4  |  17 | 84 |     17 |   17.8 |     18 |
| bbox3\_0.15\_bike\_5  |  18 | 66 |     18 |   19.3 |     21 |
| bbox3\_0.15\_bike\_6  |  19 | 21 |     19 |   19.4 |     20 |
| bbox3\_0.15\_bike\_7  |  10 | 30 |     10 |     10 |     10 |
| bbox3\_0.15\_bike\_8  |  12 | 28 |     12 |   12.1 |     13 |
| bbox3\_0.15\_bike\_9  |   7 | 26 |      7 |    7.9 |      8 |
| bbox3\_0.15\_bike\_10 |   3 |  3 |      3 |      3 |      3 |