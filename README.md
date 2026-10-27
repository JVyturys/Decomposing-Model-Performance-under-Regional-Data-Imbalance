# Decomposing Model Performance under Regional Data Imbalance

**Type:** Master's Thesis

**Author:** Jaunius Vyturys

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Katarzyna Reluga

![results](/result.png)

*Performance gap (pooled minus macro-averaged R²) per model over the realised training rows of the dominant region, Eastern Asia, across the five depletion levels.*

## Table of Content

- [Summary](#summary)
- [Working with the repo](#working-with-the-repo)
    - [Dependencies](#dependencies)
    - [Setup](#setup)
- [Reproducing results](#reproducing-results)
    - [Training code](#training-code)
    - [Evaluation code](#evaluation-code)
    - [Pretrained models](#pretrained-models)
- [Results](#results)
- [Project structure](#project-structure)

## Summary

Multi-country studies that predict ESG ratings with machine learning report accuracy only for the pooled sample. This thesis measures how strongly predictive performance declines in underrepresented regions and to what extent the resulting gap is attributable to a region's representation in the training data. The data comprise 84,027 company-years of LSEG ESG Combined Scores and Worldscope fundamentals from 9,713 entities in 16 regions. Five models from three paradigms, Random Forest, XGBoost, the FT-Transformer, TabPFN-3 and TabICLv2, are evaluated per region on a common scale. For every model, the pooled coefficient of determination exceeds the macro-averaged coefficient by 22 to 31% of the macro level. A depletion experiment reduces the training data of the dominant region, Eastern Asia, in five levels towards the size of the smallest region while the test set stays fixed, so that every change in the gap decomposes exactly into regional terms. For the tree ensembles and the in-context models, depletion removes 72 to 81% of the gap, almost entirely through the dominant region's own loss of accuracy, and 19 to 28% persists. Only one of the two in-context models, TabPFN-3, depends measurably less on the amount of training data.

**Keywords**: ESG rating prediction, regional data imbalance, performance gap decomposition, tabular foundation models, in-context learning, TabPFN, TabICL, FT-Transformer, gradient boosting, data depletion

**Full text**: [link to the published thesis, added after submission]

## Working with the repo

### Dependencies

- **Python 3.10.20.** All 97 packages are pinned to the versions of the runs of record in `requirements.txt`, among them `scikit-learn 1.7.2`, `xgboost 3.2.0`, `torch 2.12.0`, `rtdl-revisiting-models 0.0.2` (FT-Transformer), `tabpfn 8.0.3`, `tabicl 2.1.1` and `wrds 3.5.0`. The run manifests in `model_scores/` record the library versions they were produced with.
- **Hardware.** 4 × NVIDIA A40, provided by the [Humboldt Lab for Empirical and Quantitative Research](https://leqr.wiwi.hu-berlin.de/index.php). The FT-Transformer and the in-context models ran on CUDA (`device: cuda` in the manifests); `run_ftt.py` and `run_icl.py` fall back to the CPU when no GPU is available, at a much longer runtime.
- **Data access.** The raw data are pulled from Wharton Research Data Services (WRDS) and require a subscription that includes the LSEG (Refinitiv) ESG library `tr_esg`, the Refinitiv common identifiers `tr_common` and Worldscope `tr_worldscope`. The data are licensed and are not part of this repository.

### Setup

1. Clone this repository
```bash
git clone https://github.com/JVyturys/Decomposing-Model-Performance-under-Regional-Data-Imbalance.git
cd Decomposing-Model-Performance-under-Regional-Data-Imbalance
```

2. Create an environment and activate it, either with conda
```bash
conda env create -f environment.yml
conda activate tabular_benchmark
```
or with venv
```bash
python3.10 -m venv thesis-env
source thesis-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Enter your WRDS username
```bash
cp log_config.example.py log_config.py
```
Edit `log_config.py` and set `wrds_log`. The `wrds` package asks for the password on the first connection and stores it in `~/.pgpass`.

4. Create the data folders
```bash
mkdir -p data/raw data/final
```

5. Run every script as a module from the repository root, e.g.
```bash
python -m src.production.pull_raw_data.build_raw_panel
```
The scripts import `config.py`, `utils.py` and `stageguard.py` from the root, so they do not run as `python path/to/script.py`. The model runners record the current commit in their manifests and refuse to start while the working tree has uncommitted changes outside `model_scores/`.

## Reproducing results

All commands run from the repository root (see [Setup](#setup)). Each step reads the artifacts of the previous one from `data/`; paths and design constants (sample window, split shares, seed 17, region tiers, depletion anchor) are defined in `config.py`. Runtimes are the totals recorded in the run manifests on the hardware listed above.

**1. Raw data from WRDS** (`data/raw/`)
```bash
python -m src.production.pull_raw_data.build_worldscope_variable_set   # ref_ws_variables.parquet
python -m src.production.pull_raw_data.build_raw_panel                 # raw_panel.parquet
python -m src.production.pull_raw_data.build_ws_universe               # raw_ws_universe.parquet, ref_ws_nation_xwalk.parquet
python -m src.production.pull_raw_data.build_raw_esg_scores            # raw_esg_scores.parquet
```
The last two files serve only the ESG coverage figure.

**2. Final panel** (`data/raw/`, `data/final/`)
```bash
python -m src.production.from_raw.build_ref_geo_raw        # ref_geo_raw.parquet
python -m src.production.from_raw.build_ref_geo_table      # ref_geo_table.parquet
python -m src.production.from_raw.exclude_obs_wo_geo_ref   # panel.parquet
```

**3. Split, preprocessing and depletion design** (`data/final/`)
```bash
python -m src.production.from_final_panel.build_parent_reference_tables   # ref_parent_table.parquet, ref_parent_ent_type.parquet
python -m src.production.from_final_panel.build_cluster_keys              # ref_cluster_keys.parquet
python -m src.production.from_final_panel.build_split_partitions          # split.parquet
python -m src.production.from_final_panel.execute_preprocessing           # pre_processing_constants.parquet
python -m src.production.from_final_panel.build_depletion_draws           # depletion_draws.parquet, depletion_counts.parquet
```
The scripts of step 1, `build_ref_geo_raw`, `build_ref_geo_table` and `build_parent_reference_tables` query WRDS; everything from `build_cluster_keys` on runs offline.

**4. Models on the full training data** (about 18 h)
```bash
python -c "from src.production.from_final_panel.models.run_baseline import run_baseline; run_baseline('rf')"
python -c "from src.production.from_final_panel.models.run_tuned import run_tuned; run_tuned('rf'); run_tuned('xgb')"
python -c "from src.production.from_final_panel.models.run_ftt import run_ftt; run_ftt()"
python -c "from src.production.from_final_panel.models.run_icl import run_icl; run_icl('tabicl'); run_icl('tabpfn3')"
```
`run_tuned` and `run_ftt` draw 30 random-search configurations on the fit partition, select on the validation partition, refit on fit + validation and score the test partition.

**5. Depletion experiment** (about 36 h)
```bash
python -m src.production.from_final_panel.models.run_ICL_depletion_grid     # TabICLv2, TabPFN-3
python -m src.production.from_final_panel.models.run_nICL_depletion_models  # RF, XGBoost, FT-Transformer
```
Both grids enumerate the 18 frozen conditions (five levels with five, five, three, three and two draws). The in-context grid runs first, because the fitted runner checks its manifests of the same condition. Hyperparameters stay locked to the step 4 winners.

**6. Results, tables and figures** (`results/`)
```bash
python -m src.production.from_final_panel.assemble_results               # results/tables/*.csv, gap_curve_r_sq.png
python -m src.exploration.raw_data.determine_esg_coverage_asymmetry      # esg_coverage_per_region.png
python -m src.production.descriptives.build_desc_raw
python -m src.production.descriptives.build_desc_panel
python -m src.production.descriptives.build_desc_split
python -m src.production.descriptives.build_desc_features
python -m src.production.descriptives.build_desc_target_distribution
python -m src.production.descriptives.build_desc_strata
python -m src.production.descriptives.build_desc_depletion
python -m src.production.descriptives.build_desc_depletion_design       # needs build_desc_depletion
python -m src.production.descriptives.build_desc_draw_distributions
python -m src.production.descriptives.build_desc_predictions
python -m src.production.descriptives.build_desc_metrics
python -m src.production.descriptives.build_desc_assembled              # needs assemble_results
```

### Training code

The model runners are in `src/production/from_final_panel/models/`. `stageguard.py` provides each training phase with its partition (fit, validation, fit + validation, test) and applies the preprocessing constants fixed on the fit partition.

| Model | Full training data | Depletion grid |
|---|---|---|
| Random Forest (untuned) | `run_baseline.py` | |
| Random Forest, XGBoost | `run_tuned.py` | `run_nICL_depletion_models.py` |
| FT-Transformer | `run_ftt.py` | `run_nICL_depletion_models.py` |
| TabICLv2, TabPFN-3 | `run_icl.py` | `run_ICL_depletion_grid.py` |

### Evaluation code

Every runner scores the test partition and writes a manifest to `model_scores/` with hyperparameters, partition sizes, pooled and per-region metrics, library versions and the commit it ran on. The metric functions (pooled and per-region RMSE and R², macro averages) are in `utils.py`. `src/production/from_final_panel/assemble_results.py` reads all manifests and derives the pooled and macro coefficients, the gap, the share of the gap removed by depletion, the difference-in-differences decomposition and the regional contributions. The scripts in `src/production/descriptives/` produce the descriptive tables and the thesis figures.

### Pretrained models

No trained model weights are distributed. Random Forest, XGBoost and the FT-Transformer are refit by the scripts above; their selected hyperparameters are recorded in the manifests in `model_scores/`. TabICLv2 and TabPFN-3 use the pretrained checkpoints that `tabicl 2.1.1` and `tabpfn 8.0.3` download on first use. The TabPFN checkpoints are subject to the license of their provider, Prior Labs.

## Results

The table reports the main results on the fixed test partition of 25,221 rows. R²_P is the pooled coefficient of determination, R²_M its macro average over the 13 Tier-1 regions on the global scale, and G = R²_P − R²_M the gap. The depletion columns refer to the deepest level L350, at which the anchor region Eastern Asia retains about 373 training rows. Values at L350 are means over the five draws, with the range in brackets.

| Model | R²_P | R²_M | G | G / R²_M | G at L350 | Share of G removed at L350 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest (untuned) | 0.334 | 0.271 | 0.064 | 23.5% | | |
| Random Forest | 0.347 | 0.283 | 0.064 | 22.5% | 0.012 [0.010, 0.014] | 81.3% |
| XGBoost | 0.366 | 0.297 | 0.070 | 23.4% | 0.019 [0.017, 0.022] | 72.4% |
| FT-Transformer | 0.175 | 0.137 | 0.038 | 28.0% | −0.010 [−0.018, −0.005] | 127.1% |
| TabICLv2 | 0.323 | 0.247 | 0.077 | 31.2% | 0.020 [0.013, 0.027] | 74.0% |
| TabPFN-3 | 0.333 | 0.266 | 0.067 | 25.2% | 0.017 [0.013, 0.024] | 74.6% |

At L350 the pooled coefficient falls by 18 to 21% and the macro coefficient by 4 to 6% for the tree ensembles and the in-context models, so the gap closes mainly through the anchor's own loss of accuracy. The FT-Transformer's gap reverses sign.

The table is produced by `python -m src.production.from_final_panel.assemble_results`, which writes all result tables to `results/tables/` (`aggregate_undepl.csv`, `per_region_undepl.csv`, `gap_curve.csv`, `headline_changes.csv`, `did_r_sq.csv`, `gap_contributions.csv`). The descriptive tables are in `results/descriptives/` and the 25 thesis figures in `results/visualization/descriptives/`.

## Project structure

```bash
├── README.md
├── result.png                                -- header figure (gap_curve_r_sq.png)
├── requirements.txt                          -- pinned pip packages
├── environment.yml                           -- conda environment (Python 3.10.20 + requirements.txt)
├── log_config.example.py                     -- template for the WRDS username (copy to log_config.py)
├── config.py                                 -- paths, sample window, split shares, seed, region tiers, depletion constants
├── utils.py                                  -- metrics, feature cut-off, key resolution, provenance helpers
├── stageguard.py                             -- Gatekeeper: partition and preprocessing per training phase
├── data                                      -- not distributed (WRDS licence); created by the pipeline
│   ├── raw                                   -- WRDS pulls and raw reference tables
│   └── final                                 -- panel, reference tables, split, preprocessing constants, depletion draws
├── model_scores                              -- run manifests of record (YAML) and search / grid logs (JSONL)
│   ├── rf_xgb_scores                         -- Random Forest and XGBoost, full data and 18 depletion conditions
│   ├── ftt_scores                            -- FT-Transformer, full data, search log and 18 depletion conditions
│   └── icl_scores                            -- TabICLv2 and TabPFN-3, full data and 18 depletion conditions
├── results
│   ├── tables                                -- main result tables (assemble_results)
│   ├── descriptives                          -- descriptive tables (build_desc_*)
│   ├── visualization/descriptives            -- the 25 thesis figures
│   └── predictions                           -- per-observation test predictions, not distributed
└── src
    ├── production                            -- pipeline that produces every reported number
    │   ├── pull_raw_data                     -- 1  WRDS pulls (Worldscope, LSEG ESG, identifiers)
    │   ├── from_raw                          -- 2  geography reference table and final panel
    │   ├── from_final_panel                  -- 3  parent clusters, split, preprocessing, depletion draws, result assembly
    │   │   └── models                        -- 4, 5  runners for RF, XGBoost, FT-Transformer, TabICLv2, TabPFN-3
    │   └── descriptives                      -- 6  descriptive tables and thesis figures
    └── exploration                           -- diagnostics behind design decisions; only the ESG coverage figure is used in the thesis
        ├── WRDS_database                     -- library checks, identifier consistency, region names
        ├── raw_data                          -- attrition diagnostics, CUSIP link check, ESG coverage figure
        ├── final_panel                       -- clusters, geography, coverage growth, Tier-2 composition, region exclusion sensitivity
        └── split_unit                        -- missingness per region, partition target distributions
```

The manifests in `model_scores/` record the commit of the development repository each run was produced with (`git_commit`, `build_commit`, `runner_commit`). The development history is kept in a private repository; this repository contains its final state as a single snapshot.
