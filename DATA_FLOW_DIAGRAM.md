# Data Flow Diagram - Sales Forecasting Platform

This document provides comprehensive data flow diagrams for the Sales Forecasting Platform, illustrating how data moves through the system from raw inputs to final forecasts and monitoring.

---

## Table of Contents

1. [High-Level System Overview](#1-high-level-system-overview)
2. [End-to-End Pipeline Flow](#2-end-to-end-pipeline-flow)
3. [Detailed Component Diagrams](#3-detailed-component-diagrams)
4. [Data Transformation Flow](#4-data-transformation-flow)
5. [Model Architecture Data Flow](#5-model-architecture-data-flow)
6. [Monitoring & Feedback Loop](#6-monitoring--feedback-loop)
7. [Data Schema Reference](#7-data-schema-reference)

---

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         SALES FORECASTING PLATFORM - DATA FLOW                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
     │   RAW DATA   │      │   FEATURES   │      │    MODELS    │      │   OUTPUTS    │
     │   SOURCES    │─────▶│    STORE     │─────▶│   ARTIFACTS  │─────▶│  FORECASTS   │
     └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
           │                      │                     │                     │
           │                      │                     │                     │
           ▼                      ▼                     ▼                     ▼
     ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
     │ • stores.csv │      │ 55 Engineered│      │ • Global     │      │ • 30-day     │
     │ • brands.csv │      │   Features   │      │   Model      │      │   forecasts  │
     │ • sales.csv  │      │ • Temporal   │      │ • Local      │      │ • Store      │
     │ • external   │      │ • Lag/Rolling│      │   Adapters   │      │   summaries  │
     │   .csv       │      │ • Hierarchy  │      │ • Metrics    │      │ • Reports    │
     └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                                                                              │
                                                                              │
                           ┌──────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  MONITORING  │
                    │   & DRIFT    │◀──────── Feedback Loop
                    │  DETECTION   │          (Triggers Retraining)
                    └──────────────┘
```

---

## 2. End-to-End Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SEQUENTIAL PIPELINE EXECUTION                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  PHASE 1                PHASE 2                PHASE 3                PHASE 4
  DATA SETUP             PROCESSING             MODEL LIFECYCLE        OUTPUT & MONITOR
  ──────────             ──────────             ───────────────        ────────────────

┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   STEP 1    │      │     STEP 3      │      │     STEP 4      │      │     STEP 5      │
│  Generate   │      │   Ingestion &   │      │     Model       │      │   Inference &   │
│    Data     │      │    Feature      │      │    Training     │      │   Forecasting   │
│             │      │  Engineering    │      │                 │      │                 │
│ 01_generate │      │ 03_pipeline_    │      │ 04_pipeline_    │      │ 05_pipeline_    │
│ _data.py    │      │ ingestion.py    │      │ training.py     │      │ inference.py    │
└──────┬──────┘      └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
       │                      │                        │                        │
       │                      │                        │                        │
       ▼                      ▼                        ▼                        ▼
┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   STEP 2    │      │                 │      │                 │      │     STEP 6      │
│   Config    │      │   55 Features   │      │  Trained Model  │      │   Monitoring    │
│   Setup     │      │   Generated     │      │   + Adapters    │      │ & Drift Detect  │
│             │      │                 │      │                 │      │                 │
│ 02_config   │      │ feature_store/  │      │    models/      │      │ 06_pipeline_    │
│ .py         │      │ features_v1.pkl │      │ global_model    │      │ monitoring.py   │
└──────┬──────┘      └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
       │                      │                        │                        │
       │                      │                        │                        │
       ▼                      ▼                        ▼                        ▼
  ┌─────────┐          ┌───────────┐            ┌───────────┐            ┌───────────┐
  │data/    │          │416K × 55  │            │Global +   │            │Drift      │
  │*.csv    │          │records    │            │500 Local  │            │Reports    │
  └─────────┘          └───────────┘            │Adapters   │            │& Actions  │
                                                └───────────┘            └───────────┘
```

---

## 3. Detailed Component Diagrams

### 3.1 Data Generation Flow (Step 1)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              01_generate_data.py                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           SyntheticDataGenerator                                         │
│                                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   generate_     │  │   generate_     │  │   generate_     │  │   generate_     │    │
│  │   store_        │  │   brand_        │  │   sales_        │  │   external_     │    │
│  │   metadata()    │  │   metadata()    │  │   data()        │  │   features()    │    │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘    │
│           │                    │                    │                    │              │
│           ▼                    ▼                    ▼                    ▼              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  508 Stores     │  │  46 Brands      │  │  ~416K Sales    │  │  Economic &     │    │
│  │  • 500 physical │  │  • 8 categories │  │  Records        │  │  Weather Data   │    │
│  │  • 8 e-commerce │  │  • 5 price tiers│  │  • 4 years      │  │  • 8 countries  │    │
│  │  • 3 tiers      │  │  • seasonality  │  │  • 8 countries  │  │  • macro data   │    │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘    │
│           │                    │                    │                    │              │
└───────────┼────────────────────┼────────────────────┼────────────────────┼──────────────┘
            │                    │                    │                    │
            ▼                    ▼                    ▼                    ▼
       ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
       │stores   │         │brands   │         │sales    │         │external │
       │.csv     │         │.csv     │         │.csv     │         │.csv     │
       └─────────┘         └─────────┘         └─────────┘         └─────────┘
            │                    │                    │                    │
            └────────────────────┴────────────────────┴────────────────────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │ data/       │
                                   │ directory   │
                                   └─────────────┘
```

### 3.2 Data Ingestion & Feature Engineering Flow (Step 3)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            03_pipeline_ingestion.py                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

        INPUT                      VALIDATION                   FEATURE ENGINEERING
        ─────                      ──────────                   ───────────────────

  ┌─────────────┐            ┌─────────────────────┐
  │ stores.csv  │───────┐    │    DataValidator    │
  │ (508 rows)  │       │    │                     │
  └─────────────┘       │    │ • Schema validation │
                        │    │ • Outlier detection │
  ┌─────────────┐       ├───▶│   (z-score > 3.0)   │
  │ brands.csv  │───────┤    │ • Missing values    │
  │ (46 rows)   │       │    │ • Duplicates check  │
  └─────────────┘       │    │ • Negative sales    │
                        │    │                     │
  ┌─────────────┐       │    └──────────┬──────────┘
  │ sales.csv   │───────┤              │
  │ (~416K rows)│       │              │ Validated Data
  └─────────────┘       │              ▼
                        │    ┌─────────────────────┐
  ┌─────────────┐       │    │  FeatureEngineer    │
  │ external.csv│───────┘    │                     │
  │ (1460 rows) │            │ Creates 55 Features │
  └─────────────┘            │                     │
                             └──────────┬──────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │ TEMPORAL (15)   │ │ LAG/ROLLING (16)│ │ HIERARCHY (10)  │
          │                 │ │                 │ │                 │
          │ • day_of_week   │ │ • lag_7d        │ │ • store_size    │
          │ • month         │ │ • lag_14d       │ │ • location_type │
          │ • quarter       │ │ • lag_30d       │ │ • tier          │
          │ • is_weekend    │ │ • lag_365d      │ │ • country       │
          │ • sin/cos       │ │ • rolling_mean  │ │ • brand_category│
          │   seasonality   │ │ • rolling_std   │ │ • price_tier    │
          │ • days_to_event │ │ • rolling_max   │ │ • store_cluster │
          └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                   │                   │                   │
                   └───────────────────┼───────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
          ┌─────────────────┐               ┌─────────────────────┐
          │ EXTERNAL (5)    │               │ PROMOTIONAL (4)     │
          │                 │               │                     │
          │ • gdp_growth    │               │ • promotion_flag    │
          │ • tourism_index │               │ • promotion_discount│
          │ • currency_rate │               │ • num_transactions  │
          │ • temperature   │               │ • avg_txn_value     │
          │ • is_holiday    │               │                     │
          └────────┬────────┘               └──────────┬──────────┘
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    OUTPUT             │
                         │                       │
                         │  feature_store/       │
                         │  ├── features_v1.pkl  │
                         │  │   (416K × 55)      │
                         │  └── metadata_v1.json │
                         │      (feature defs)   │
                         └───────────────────────┘
```

### 3.3 Model Training Flow (Step 4)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              04_pipeline_training.py                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

         INPUT                    DATA SPLIT                    MODEL TRAINING
         ─────                    ──────────                    ──────────────

  ┌─────────────────┐       ┌─────────────────────┐
  │ feature_store/  │       │    DataSplitter     │
  │ features_v1.pkl │──────▶│  (Time-Based Split) │
  │ (416K × 55)     │       │                     │
  └─────────────────┘       └──────────┬──────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
             ┌───────────┐      ┌───────────┐      ┌───────────┐
             │   TRAIN   │      │VALIDATION │      │   TEST    │
             │   (80%)   │      │   (10%)   │      │   (10%)   │
             │ ~333K rows│      │ ~42K rows │      │ ~42K rows │
             └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
                   │                  │                  │
                   │                  │                  │
                   ▼                  │                  │
  ┌────────────────────────────────┐  │                  │
  │         TIER 1: GLOBAL         │  │                  │
  │      FOUNDATION MODEL          │  │                  │
  │                                │  │                  │
  │  ┌──────────────────────────┐  │  │                  │
  │  │   Gradient Boosting      │  │  │                  │
  │  │   • n_estimators: 200    │  │  │                  │
  │  │   • learning_rate: 0.05  │  │  │                  │
  │  │   • max_depth: 7         │  │  │                  │
  │  │   • subsample: 0.8       │  │  │                  │
  │  └──────────────────────────┘  │  │                  │
  │                                │  │                  │
  │  INPUT: All 55 features        │  │                  │
  │  OUTPUT: Global predictions    │  │                  │
  └────────────────┬───────────────┘  │                  │
                   │                  │                  │
                   ▼                  │                  │
  ┌────────────────────────────────┐  │                  │
  │         TIER 2: HIERARCHY      │  │                  │
  │        EMBEDDINGS              │  │                  │
  │                                │  │                  │
  │  • Country embeddings (8)      │  │                  │
  │  • Brand embeddings (46)       │  │                  │
  │  • Store clusters (10 K-Means) │  │                  │
  │  • Location type encoding      │  │                  │
  └────────────────┬───────────────┘  │                  │
                   │                  │                  │
                   ▼                  ▼                  │
  ┌────────────────────────────────────────────────────┐ │
  │          TIER 3: LOCAL ADAPTATION                  │ │
  │                                                    │ │
  │  For each store with 180+ days data:               │ │
  │                                                    │ │
  │  ┌──────────────────────────────────────────────┐  │ │
  │  │  Final = GlobalPred + (Shrinkage × Conf ×    │  │ │
  │  │                        LocalResidual)        │  │ │
  │  │                                              │  │ │
  │  │  Shrinkage = 0.7                             │  │ │
  │  │  Confidence = min(DataPoints / 1095, 1.0)   │  │ │
  │  └──────────────────────────────────────────────┘  │ │
  │                                                    │ │
  └────────────────────────┬───────────────────────────┘ │
                           │                             │
                           │         ┌───────────────────┘
                           │         │
                           ▼         ▼
                    ┌─────────────────────┐
                    │     EVALUATION      │
                    │                     │
                    │  • MAPE: ~8%        │
                    │  • MAE: ~560        │
                    │  • RMSE: ~771       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │       OUTPUT          │
                    │                       │
                    │  models/              │
                    │  ├── global_model_    │
                    │  │   v1.pkl           │
                    │  ├── local_adapters_  │
                    │  │   v1.pkl           │
                    │  └── training_        │
                    │      results_v1.json  │
                    └───────────────────────┘
```

### 3.4 Inference & Forecasting Flow (Step 5)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              05_pipeline_inference.py                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

          INPUTS                    PROCESSING                    OUTPUTS
          ──────                    ──────────                    ───────

  ┌─────────────────┐       ┌─────────────────────────────────────────────────┐
  │ models/         │       │              ForecastGenerator                   │
  │ global_model_   │──┐    │                                                  │
  │ v1.pkl          │  │    │  ┌─────────────────────────────────────────┐    │
  └─────────────────┘  │    │  │     For each of 508 stores:             │    │
                       │    │  │                                         │    │
  ┌─────────────────┐  │    │  │  1. Load historical data                │    │
  │ models/         │  │    │  │  2. Generate 30 future dates            │    │
  │ local_adapters_ │──┼───▶│  │  3. Create feature records:             │    │
  │ v1.pkl          │  │    │  │     • Temporal (from date)              │    │
  │ (500 stores)    │  │    │  │     • Static (store metadata)           │    │
  └─────────────────┘  │    │  │     • Lag (from recent history)         │    │
                       │    │  │     • External (macro indicators)       │    │
  ┌─────────────────┐  │    │  │  4. Generate predictions:               │    │
  │ feature_store/  │  │    │  │     • Global model base                 │    │
  │ features_v1.pkl │──┘    │  │     • + Local adapter adjustment        │    │
  │ (recent data)   │       │  │  5. Calculate 80% confidence bounds     │    │
  └─────────────────┘       │  │                                         │    │
                            │  └─────────────────────────────────────────┘    │
                            │                                                  │
                            └──────────────────────┬──────────────────────────┘
                                                   │
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
                    ▼                              ▼                              ▼
          ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
          │ forecasts_next_     │      │ forecast_           │      │ forecast_           │
          │ 30days.csv          │      │ summary.csv         │      │ report.txt          │
          │                     │      │                     │      │                     │
          │ • 15,240 rows       │      │ • 508 rows          │      │ • Summary stats     │
          │ • date              │      │ • store_id          │      │ • Aggregates by     │
          │ • store_id          │      │ • brand_id          │      │   country/brand     │
          │ • brand_id          │      │ • country           │      │ • Top/bottom        │
          │ • country           │      │ • total_forecast    │      │   performers        │
          │ • predicted_sales   │      │ • avg_daily_        │      │                     │
          │ • lower_bound       │      │   forecast          │      │                     │
          │ • upper_bound       │      │ • prediction_type   │      │                     │
          │ • confidence_level  │      │                     │      │                     │
          │ • prediction_type   │      │                     │      │                     │
          └─────────────────────┘      └─────────────────────┘      └─────────────────────┘
                    │                              │                              │
                    └──────────────────────────────┴──────────────────────────────┘
                                                   │
                                                   ▼
                                          ┌───────────────┐
                                          │   outputs/    │
                                          │   directory   │
                                          └───────────────┘
```

### 3.5 Monitoring & Drift Detection Flow (Step 6)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              06_pipeline_monitoring.py                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

          INPUTS                   DRIFT DETECTION                  OUTPUTS
          ──────                   ───────────────                  ───────

  ┌─────────────────┐       ┌───────────────────────────────────────────────────────────┐
  │ outputs/        │       │                  DriftDetector                             │
  │ forecasts_next_ │───┐   │                                                           │
  │ 30days.csv      │   │   │   ┌─────────────────────────────────────────────────────┐ │
  └─────────────────┘   │   │   │           METHOD 1: MAPE-Based Drift                │ │
                        │   │   │                 (MAPEMonitor)                        │ │
  ┌─────────────────┐   │   │   │                                                     │ │
  │ feature_store/  │   │   │   │   Rolling 7-day MAPE per store                      │ │
  │ features_v1.pkl │───┼──▶│   │                                                     │ │
  │ (actuals)       │   │   │   │   Thresholds:                                       │ │
  └─────────────────┘   │   │   │   • Minor:    5-8%  → Alert only                    │ │
                        │   │   │   • Moderate: 8-12% → Retrain cluster               │ │
  ┌─────────────────┐   │   │   │   • Severe:   >12%  → Emergency retrain             │ │
  │ models/         │   │   │   │                                                     │ │
  │ training_       │───┘   │   └─────────────────────────────────────────────────────┘ │
  │ results_v1.json │       │                                                           │
  └─────────────────┘       │   ┌─────────────────────────────────────────────────────┐ │
                            │   │           METHOD 2: PSI-Based Drift                 │ │
                            │   │              (PSICalculator)                         │ │
                            │   │                                                     │ │
                            │   │   Population Stability Index per feature            │ │
                            │   │                                                     │ │
                            │   │   Thresholds:                                       │ │
                            │   │   • PSI < 0.10:  No change                          │ │
                            │   │   • 0.10 - 0.25: Monitor                            │ │
                            │   │   • PSI > 0.25:  Significant drift                  │ │
                            │   │                                                     │ │
                            │   └─────────────────────────────────────────────────────┘ │
                            │                                                           │
                            │   ┌─────────────────────────────────────────────────────┐ │
                            │   │           METHOD 3: CUSUM-Based Drift               │ │
                            │   │               (CUSUMMonitor)                         │ │
                            │   │                                                     │ │
                            │   │   Cumulative sum of forecast errors                 │ │
                            │   │   Detects gradual degradation over time             │ │
                            │   │                                                     │ │
                            │   └─────────────────────────────────────────────────────┘ │
                            │                                                           │
                            └───────────────────────────┬───────────────────────────────┘
                                                        │
                                                        │
                    ┌───────────────────────────────────┼───────────────────────────────┐
                    │                                   │                               │
                    ▼                                   ▼                               ▼
          ┌─────────────────────┐          ┌─────────────────────┐          ┌─────────────────────┐
          │ drift_detection_    │          │ retraining_         │          │ Store Health        │
          │ report.json         │          │ recommendations.txt │          │ Dashboard           │
          │                     │          │                     │          │ (implicit)          │
          │ • Overall status    │          │ • Action items      │          │                     │
          │ • MAPE analysis     │          │ • Priority stores   │          │ • Healthy: 53       │
          │ • PSI analysis      │          │ • Recommended       │          │ • Warning: 178      │
          │ • CUSUM analysis    │          │   actions           │          │ • Critical: 269     │
          │ • Store breakdown   │          │                     │          │                     │
          └─────────────────────┘          └─────────────────────┘          └─────────────────────┘
                    │                                   │                               │
                    └───────────────────────────────────┴───────────────────────────────┘
                                                        │
                                                        ▼
                                               ┌───────────────┐
                                               │  monitoring/  │
                                               │   directory   │
                                               └───────────────┘
```

---

## 4. Data Transformation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         DATA TRANSFORMATION JOURNEY                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

    RAW DATA                 FEATURES                  MODEL                    OUTPUT
    ────────                 ────────                  ─────                    ──────

┌─────────────┐         ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   SALES     │         │  TEMPORAL   │         │   GLOBAL    │         │  FORECAST   │
│   RECORD    │         │  FEATURES   │         │   SCORE     │         │   VALUE     │
│             │         │             │         │             │         │             │
│ date        │────────▶│ day_of_week │         │             │         │ predicted_  │
│ store_id    │         │ month       │         │ Base        │────────▶│ sales       │
│ sales_amount│         │ is_weekend  │────────▶│ Prediction  │         │             │
│ promotion_  │         │ seasonality │         │             │         │ lower_bound │
│ flag        │         │             │         │             │         │ upper_bound │
└─────────────┘         └─────────────┘         └─────────────┘         └─────────────┘
      │                       │                       │                       │
      │                       │                       │                       │
      ▼                       ▼                       ▼                       ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   STORE     │         │    LAG      │         │   LOCAL     │         │  FORECAST   │
│   METADATA  │         │  FEATURES   │         │   ADAPTER   │         │   TYPE      │
│             │         │             │         │             │         │             │
│ store_size  │────────▶│ lag_7d      │         │ Store-      │────────▶│ adapted     │
│ location    │         │ lag_14d     │────────▶│ Specific    │         │    OR       │
│ tier        │         │ lag_30d     │         │ Adjustment  │         │ global_only │
│ is_ecommerce│         │ lag_365d    │         │             │         │             │
└─────────────┘         └─────────────┘         └─────────────┘         └─────────────┘
      │                       │                       │                       │
      │                       │                       │                       │
      ▼                       ▼                       ▼                       ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   BRAND     │         │  ROLLING    │         │  BAYESIAN   │         │ CONFIDENCE  │
│   METADATA  │         │   STATS     │         │  SHRINKAGE  │         │   LEVEL     │
│             │         │             │         │             │         │             │
│ category    │────────▶│ mean_7d     │         │ Shrinkage   │────────▶│ 80%         │
│ price_tier  │         │ std_14d     │────────▶│ × Confidence│         │ prediction  │
│ seasonality │         │ max_30d     │         │ × Residual  │         │ interval    │
│             │         │             │         │             │         │             │
└─────────────┘         └─────────────┘         └─────────────┘         └─────────────┘
      │                       │
      │                       │
      ▼                       ▼
┌─────────────┐         ┌─────────────┐
│  EXTERNAL   │         │ HIERARCHY   │
│    DATA     │         │  FEATURES   │
│             │         │             │
│ gdp_growth  │────────▶│ country_emb │
│ tourism_idx │         │ brand_emb   │
│ currency    │         │ cluster     │
│ temperature │         │             │
└─────────────┘         └─────────────┘

```

---

## 5. Model Architecture Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     3-TIER HIERARCHICAL MODEL ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT FEATURES (55)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                              │
                    ▼                                              ▼
    ┌───────────────────────────────────┐      ┌───────────────────────────────────┐
    │        ALL FEATURES (55)          │      │     STORE HISTORICAL DATA         │
    │                                   │      │                                   │
    │  Temporal (15) + Lag (4) +        │      │  Sales history for specific       │
    │  Rolling (12) + Hierarchy (10) +  │      │  store (if 180+ days available)   │
    │  External (5) + Promotional (4) + │      │                                   │
    │  Transaction (2) + Other (3)      │      │                                   │
    └─────────────────┬─────────────────┘      └─────────────────┬─────────────────┘
                      │                                          │
                      ▼                                          │
    ┌─────────────────────────────────────────────────────────┐  │
    │                                                         │  │
    │                 TIER 1: GLOBAL MODEL                    │  │
    │                                                         │  │
    │   ┌───────────────────────────────────────────────────┐ │  │
    │   │           Gradient Boosting Regressor             │ │  │
    │   │                                                   │ │  │
    │   │   Trained on ALL 500 stores combined              │ │  │
    │   │   Learns universal sales patterns                 │ │  │
    │   │   Base prediction for any store                   │ │  │
    │   │                                                   │ │  │
    │   │   Hyperparameters:                                │ │  │
    │   │   n_estimators=200, learning_rate=0.05,           │ │  │
    │   │   max_depth=7, subsample=0.8                      │ │  │
    │   └───────────────────────────────────────────────────┘ │  │
    │                                                         │  │
    └───────────────────────────┬─────────────────────────────┘  │
                                │                                │
                                ▼                                │
    ┌─────────────────────────────────────────────────────────┐  │
    │                                                         │  │
    │              TIER 2: HIERARCHY EMBEDDINGS               │  │
    │                                                         │  │
    │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │  │
    │   │  COUNTRY    │  │   BRAND     │  │   STORE     │    │  │
    │   │  EMBEDDING  │  │  EMBEDDING  │  │  CLUSTER    │    │  │
    │   │             │  │             │  │             │    │  │
    │   │  8 countries│  │  46 brands  │  │  10 clusters│    │  │
    │   │  regional   │  │  category & │  │  K-Means    │    │  │
    │   │  patterns   │  │  price tier │  │  grouping   │    │  │
    │   └─────────────┘  └─────────────┘  └─────────────┘    │  │
    │                                                         │  │
    └───────────────────────────┬─────────────────────────────┘  │
                                │                                │
                                ▼                                │
                     GlobalPrediction                            │
                                │                                │
                                │         ┌──────────────────────┘
                                │         │
                                ▼         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │                     TIER 3: LOCAL ADAPTATION                            │
    │                                                                         │
    │   ┌───────────────────────────────────────────────────────────────────┐ │
    │   │   For stores with 180+ days of history:                          │ │
    │   │                                                                   │ │
    │   │   LocalResidual = Actual - GlobalPrediction (on training data)   │ │
    │   │                                                                   │ │
    │   │   Confidence = min(NumDataPoints / 1095, 1.0)                    │ │
    │   │                                                                   │ │
    │   │   FinalPrediction = GlobalPrediction +                           │ │
    │   │                     (0.7 × Confidence × MeanResidual)            │ │
    │   │                                                                   │ │
    │   └───────────────────────────────────────────────────────────────────┘ │
    │                                                                         │
    │   For stores with < 180 days: Use GlobalPrediction only                │
    │                                                                         │
    └───────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
                                  ┌───────────────────┐
                                  │ FINAL FORECAST    │
                                  │                   │
                                  │ • Point estimate  │
                                  │ • Lower bound     │
                                  │ • Upper bound     │
                                  │ • Prediction type │
                                  └───────────────────┘
```

---

## 6. Monitoring & Feedback Loop

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CONTINUOUS MONITORING & FEEDBACK LOOP                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                              ┌────────────────────────┐
                              │    PRODUCTION DATA     │
                              │                        │
                              │  New daily sales       │
                              │  from 508 stores       │
                              └───────────┬────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DRIFT DETECTION ENGINE                                      │
│                                                                                          │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐          │
│   │   MAPE MONITOR      │   │   PSI CALCULATOR    │   │   CUSUM MONITOR     │          │
│   │                     │   │                     │   │                     │          │
│   │  Error-based        │   │  Distribution       │   │  Cumulative         │          │
│   │  drift detection    │   │  shift detection    │   │  sum analysis       │          │
│   │                     │   │                     │   │                     │          │
│   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │          │
│   │  │ MAPE < 5%     │──┼───┼─▶│ PSI < 0.10    │──┼───┼─▶│ Within limits │──┼───┐      │
│   │  │ HEALTHY       │  │   │  │ STABLE        │  │   │  │ NO DRIFT      │  │   │      │
│   │  └───────────────┘  │   │  └───────────────┘  │   │  └───────────────┘  │   │      │
│   │                     │   │                     │   │                     │   │      │
│   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │      │
│   │  │ 5% < MAPE < 8%│──┼───┼─▶│ 0.10 < PSI <  │──┼───┼─▶│ Approaching   │──┼───┤      │
│   │  │ WARNING       │  │   │  │ 0.25 MONITOR  │  │   │  │ threshold     │  │   │      │
│   │  └───────────────┘  │   │  └───────────────┘  │   │  └───────────────┘  │   │      │
│   │                     │   │                     │   │                     │   │      │
│   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │   │      │
│   │  │ MAPE > 8%     │──┼───┼─▶│ PSI > 0.25    │──┼───┼─▶│ Exceeded      │──┼───┤      │
│   │  │ CRITICAL      │  │   │  │ DRIFTED       │  │   │  │ DRIFT         │  │   │      │
│   │  └───────────────┘  │   │  └───────────────┘  │   │  └───────────────┘  │   │      │
│   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘   │      │
│                                                                                 │      │
└─────────────────────────────────────────────────────────────────────────────────┼──────┘
                                                                                  │
                                     ┌────────────────────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────────┐
                    │          DECISION ENGINE           │
                    │                                    │
                    │   Based on combined drift signals: │
                    └────────────────┬───────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
   ┌────────────────┐     ┌────────────────────┐    ┌────────────────────┐
   │  NO ACTION     │     │   CLUSTER RETRAIN  │    │   GLOBAL RETRAIN   │
   │                │     │                    │    │                    │
   │  All healthy   │     │  10-50 stores      │    │  All 500 stores    │
   │  Continue      │     │  in cluster        │    │  Full model        │
   │  monitoring    │     │  affected          │    │  rebuild           │
   │                │     │                    │    │                    │
   │  Cost: $0      │     │  Cost: Medium      │    │  Cost: High        │
   │  Frequency:    │     │  Frequency:        │    │  Frequency:        │
   │  Continuous    │     │  Weekly            │    │  Monthly           │
   └────────────────┘     └─────────┬──────────┘    └──────────┬─────────┘
                                    │                          │
                                    └────────────┬─────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────┐
                                  │     RETRAIN PIPELINE      │
                                  │                           │
                                  │  1. Load updated data     │
                                  │  2. Re-run ingestion      │
                                  │  3. Re-train model        │
                                  │  4. Validate metrics      │
                                  │  5. Deploy new model      │
                                  │                           │
                                  └─────────────┬─────────────┘
                                                │
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         │                                             │
                         ▼                                             ▼
              ┌────────────────────┐                      ┌─────────────────────┐
              │  Updated Model     │                      │  Reset Monitoring   │
              │  Deployed          │                      │  Baselines          │
              │                    │                      │                     │
              │  models/           │                      │  New reference      │
              │  global_model_v2   │                      │  distributions      │
              │  local_adapters_v2 │                      │  set for PSI        │
              └────────────────────┘                      └─────────────────────┘
```

---

## 7. Data Schema Reference

### 7.1 Input Data Schemas

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  INPUT DATA SCHEMAS                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─ stores.csv ──────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  store_id        │ VARCHAR(20)  │ PK │ Unique store identifier (STORE_001, ECOM_UAE) │
│  store_name      │ VARCHAR(100) │    │ Human-readable name                           │
│  brand_id        │ VARCHAR(20)  │ FK │ Link to brands.csv                            │
│  country         │ VARCHAR(50)  │    │ Operating country                             │
│  location_type   │ VARCHAR(20)  │    │ mall / street / airport / online              │
│  store_size_sqm  │ FLOAT        │    │ Physical store size                           │
│  tier            │ VARCHAR(20)  │    │ data_rich / data_medium / data_poor           │
│  history_days    │ INTEGER      │    │ Days of historical data available             │
│  opening_date    │ DATE         │    │ Store opening date                            │
│  is_ecommerce    │ BOOLEAN      │    │ True for online stores                        │
│                                                                                        │
│  Rows: 508 (500 physical + 8 e-commerce)                                              │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─ brands.csv ──────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  brand_id             │ VARCHAR(20)  │ PK │ Unique brand identifier                   │
│  brand_name           │ VARCHAR(100) │    │ Brand name                                │
│  category             │ VARCHAR(50)  │    │ Luxury Watches / Fashion / Cosmetics etc. │
│  price_tier           │ VARCHAR(20)  │    │ ultra_luxury / luxury / premium etc.      │
│  avg_price_point      │ FLOAT        │    │ Average item price                        │
│  seasonality_strength │ FLOAT        │    │ 0.0-1.0 seasonality factor                │
│                                                                                        │
│  Rows: 46                                                                             │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─ sales.csv ───────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  date                │ DATE         │    │ Transaction date                           │
│  store_id            │ VARCHAR(20)  │ FK │ Link to stores.csv                         │
│  brand_id            │ VARCHAR(20)  │ FK │ Link to brands.csv                         │
│  country             │ VARCHAR(50)  │    │ Country of sale                            │
│  sales_amount        │ FLOAT        │    │ Daily sales value                          │
│  num_transactions    │ INTEGER      │    │ Number of transactions                     │
│  avg_transaction_val │ FLOAT        │    │ Average transaction value                  │
│  promotion_flag      │ BOOLEAN      │    │ Whether promotion was active               │
│  promotion_discount  │ FLOAT        │    │ Discount percentage (0-1)                  │
│                                                                                        │
│  Rows: ~416,000 (4 years × 508 stores, varying by store age)                         │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─ external.csv ────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  date               │ DATE         │    │ Date                                        │
│  country            │ VARCHAR(50)  │    │ Country                                     │
│  gdp_growth         │ FLOAT        │    │ GDP growth rate                             │
│  tourism_index      │ FLOAT        │    │ Tourism activity index                      │
│  currency_rate_usd  │ FLOAT        │    │ Exchange rate to USD                        │
│  temperature_celsius│ FLOAT        │    │ Average temperature                         │
│  is_public_holiday  │ BOOLEAN      │    │ Public holiday flag                         │
│                                                                                        │
│  Rows: ~1,460 (4 years × 365 days, per country)                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Feature Store Schema

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               FEATURE STORE SCHEMA (55 Features)                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─ features_v1.pkl ─────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  TEMPORAL FEATURES (15)                                                               │
│  ─────────────────────                                                                │
│  day_of_week        │ INT    │ 0-6 (Monday-Sunday)                                   │
│  day_of_month       │ INT    │ 1-31                                                  │
│  month              │ INT    │ 1-12                                                  │
│  quarter            │ INT    │ 1-4                                                   │
│  week_of_year       │ INT    │ 1-52                                                  │
│  is_weekend         │ BOOL   │ Saturday/Sunday flag                                  │
│  is_month_start     │ BOOL   │ First 3 days of month                                 │
│  is_month_end       │ BOOL   │ Last 3 days of month                                  │
│  sin_day_of_year    │ FLOAT  │ Cyclic encoding                                       │
│  cos_day_of_year    │ FLOAT  │ Cyclic encoding                                       │
│  sin_week_of_year   │ FLOAT  │ Cyclic encoding                                       │
│  cos_week_of_year   │ FLOAT  │ Cyclic encoding                                       │
│  days_to_black_fri  │ INT    │ Days until Black Friday                               │
│  days_to_new_year   │ INT    │ Days until New Year                                   │
│  days_to_ramadan    │ INT    │ Days until Ramadan                                    │
│                                                                                        │
│  LAG FEATURES (4)                                                                     │
│  ─────────────────                                                                    │
│  sales_lag_7d       │ FLOAT  │ Sales 7 days ago                                      │
│  sales_lag_14d      │ FLOAT  │ Sales 14 days ago                                     │
│  sales_lag_30d      │ FLOAT  │ Sales 30 days ago                                     │
│  sales_lag_365d     │ FLOAT  │ Sales 365 days ago                                    │
│                                                                                        │
│  ROLLING STATISTICS (12)                                                              │
│  ──────────────────────                                                               │
│  rolling_mean_7d    │ FLOAT  │ 7-day rolling mean                                    │
│  rolling_std_7d     │ FLOAT  │ 7-day rolling std                                     │
│  rolling_max_7d     │ FLOAT  │ 7-day rolling max                                     │
│  rolling_min_7d     │ FLOAT  │ 7-day rolling min                                     │
│  rolling_mean_14d   │ FLOAT  │ 14-day rolling mean                                   │
│  rolling_std_14d    │ FLOAT  │ 14-day rolling std                                    │
│  rolling_max_14d    │ FLOAT  │ 14-day rolling max                                    │
│  rolling_min_14d    │ FLOAT  │ 14-day rolling min                                    │
│  rolling_mean_30d   │ FLOAT  │ 30-day rolling mean                                   │
│  rolling_std_30d    │ FLOAT  │ 30-day rolling std                                    │
│  rolling_max_30d    │ FLOAT  │ 30-day rolling max                                    │
│  rolling_min_30d    │ FLOAT  │ 30-day rolling min                                    │
│                                                                                        │
│  HIERARCHICAL FEATURES (10)                                                           │
│  ─────────────────────────                                                            │
│  store_size_sqm     │ FLOAT  │ Store physical size                                   │
│  location_encoded   │ INT    │ Location type encoding                                │
│  tier_encoded       │ INT    │ Data tier encoding                                    │
│  country_encoded    │ INT    │ Country encoding                                      │
│  category_encoded   │ INT    │ Brand category encoding                               │
│  price_tier_encoded │ INT    │ Price tier encoding                                   │
│  avg_price_point    │ FLOAT  │ Average price point                                   │
│  seasonality_str    │ FLOAT  │ Brand seasonality strength                            │
│  store_cluster      │ INT    │ K-means cluster ID (0-9)                              │
│  is_ecommerce       │ BOOL   │ E-commerce flag                                       │
│                                                                                        │
│  EXTERNAL FEATURES (5)                                                                │
│  ────────────────────                                                                 │
│  gdp_growth         │ FLOAT  │ GDP growth rate                                       │
│  tourism_index      │ FLOAT  │ Tourism activity                                      │
│  currency_rate      │ FLOAT  │ Currency exchange rate                                │
│  temperature        │ FLOAT  │ Temperature                                           │
│  is_public_holiday  │ BOOL   │ Holiday flag                                          │
│                                                                                        │
│  PROMOTIONAL FEATURES (4)                                                             │
│  ───────────────────────                                                              │
│  promotion_flag     │ BOOL   │ Promotion active                                      │
│  promotion_discount │ FLOAT  │ Discount percentage                                   │
│  num_transactions   │ INT    │ Transaction count                                     │
│  avg_txn_value      │ FLOAT  │ Average transaction value                             │
│                                                                                        │
│  TARGET & IDENTIFIERS (5)                                                             │
│  ───────────────────────                                                              │
│  date               │ DATE   │ Transaction date                                      │
│  store_id           │ STR    │ Store identifier                                      │
│  brand_id           │ STR    │ Brand identifier                                      │
│  country            │ STR    │ Country                                               │
│  sales_amount       │ FLOAT  │ TARGET: Daily sales                                   │
│                                                                                        │
│  Total: 416,000 rows × 55 columns                                                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Output Data Schemas

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  OUTPUT DATA SCHEMAS                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─ forecasts_next_30days.csv ───────────────────────────────────────────────────────────┐
│                                                                                        │
│  date             │ DATE         │    │ Forecast date                                 │
│  store_id         │ VARCHAR(20)  │    │ Store identifier                              │
│  brand_id         │ VARCHAR(20)  │    │ Brand identifier                              │
│  country          │ VARCHAR(50)  │    │ Country                                       │
│  predicted_sales  │ FLOAT        │    │ Point forecast                                │
│  lower_bound      │ FLOAT        │    │ 80% CI lower bound                            │
│  upper_bound      │ FLOAT        │    │ 80% CI upper bound                            │
│  confidence_level │ FLOAT        │    │ Prediction confidence (0-1)                   │
│  prediction_type  │ VARCHAR(20)  │    │ adapted / global_only                         │
│                                                                                        │
│  Rows: 15,240 (508 stores × 30 days)                                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─ forecast_summary.csv ────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  store_id          │ VARCHAR(20)  │ PK │ Store identifier                             │
│  brand_id          │ VARCHAR(20)  │    │ Brand identifier                             │
│  country           │ VARCHAR(50)  │    │ Country                                      │
│  total_forecast    │ FLOAT        │    │ 30-day total forecast                        │
│  avg_daily_forecast│ FLOAT        │    │ Average daily forecast                       │
│  prediction_type   │ VARCHAR(20)  │    │ adapted / global_only                        │
│                                                                                        │
│  Rows: 508                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─ drift_detection_report.json ─────────────────────────────────────────────────────────┐
│                                                                                        │
│  {                                                                                     │
│    "timestamp": "2026-01-15T10:30:00Z",                                               │
│    "overall_status": "DRIFT_DETECTED",                                                │
│    "recommended_action": "RETRAIN_CLUSTER",                                           │
│    "mape_analysis": {                                                                 │
│      "healthy_stores": 53,                                                            │
│      "warning_stores": 178,                                                           │
│      "critical_stores": 269,                                                          │
│      "critical_store_ids": ["STORE_402", "STORE_409", ...]                           │
│    },                                                                                 │
│    "psi_analysis": {                                                                  │
│      "features_drifted": 8,                                                           │
│      "features_stable": 38,                                                           │
│      "drifted_features": ["rolling_mean_7d", "lag_7d", ...]                          │
│    },                                                                                 │
│    "cusum_analysis": {                                                                │
│      "drift_detected": true,                                                          │
│      "cumulative_sum": 4.87,                                                          │
│      "threshold": 5.0                                                                 │
│    }                                                                                  │
│  }                                                                                    │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This data flow diagram document illustrates the complete journey of data through the Sales Forecasting Platform:

1. **Data Generation**: Synthetic data creation for 508 stores, 46 brands, ~416K sales records
2. **Data Ingestion**: Validation and 55-feature engineering pipeline
3. **Model Training**: 3-tier hierarchical architecture (Global + Hierarchy + Local)
4. **Inference**: 30-day forecasting with confidence intervals
5. **Monitoring**: Multi-method drift detection (MAPE, PSI, CUSUM) with feedback loops

The architecture enables:
- **95% cost savings** vs training individual store models
- **Cold-start handling** through hierarchical pooling
- **Automated retraining** triggered by drift detection
- **Scalability** to 5,000+ stores with distributed computing
