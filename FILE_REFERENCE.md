#  File Reference Guide

## Complete Module Breakdown

---

##  Overview

This project contains **11 files** organized into **6 executable scripts**, **3 documentation files**, and **2 configuration files**.

---

##  Configuration Files

### `config.py`
**Purpose**: Central configuration module  
**What it does**:
- Defines all configuration classes (DataConfig, ModelConfig, DriftConfig, InferenceConfig)
- Sets hyperparameters, thresholds, and file paths
- Creates directory structure

**Key settings you can modify**:
```python
# Model hyperparameters
n_estimators = 200           # Reduce to 100 for faster training
learning_rate = 0.05         # Lower = more accurate but slower
max_depth = 7                # Tree depth

# Drift thresholds
mape_alert_threshold = 8.0   # MAPE % to trigger alerts
psi_threshold = 0.25         # Feature drift threshold

# Feature engineering
lag_features = [7, 14, 30, 365]      # Historical lags to use
rolling_windows = [7, 14, 30]        # Rolling window sizes
```

**When to run**: Imported by all other modules (runs automatically)

---

### `config.json`
**Purpose**: Saved configuration in JSON format  
**Created by**: `02_config.py`  
**What it contains**: All settings from config.py in JSON format  
**When to modify**: After running `02_config.py`, you can edit this file to change settings

---

##  Executable Scripts

### 1. `01_generate_data.py`
**Purpose**: Generate synthetic sales data  
**Run order**: 1st  
**Execution time**: 30-60 seconds  
**Command**: `python 01_generate_data.py`

**What it creates**:
```
data/
├── stores.csv      # 508 stores (500 physical + 8 e-commerce)
├── brands.csv      # 46 brands across 8 categories
├── sales.csv       # ~416,000 daily sales records
├── external.csv    # Economic/weather features
└── summary.json    # Dataset statistics
```

**Key Features**:
- **Stores**: 3 tiers (data_rich: 3+ years, data_medium: 1-3 years, data_poor: <1 year)
- **Brands**: Luxury watches, fashion, cosmetics, jewelry, etc.
- **Countries**: UAE, Saudi Arabia, Kuwait, Qatar, Bahrain, Oman, Egypt, Jordan
- **Patterns**: Seasonal trends, weekend effects, promotions, holidays

**Class Structure**:
```python
SyntheticDataGenerator
├── generate_store_metadata()     # Create store info
├── generate_brand_metadata()     # Create brand info
├── generate_sales_data()         # Generate daily sales with patterns
├── generate_external_features()  # Economic/weather data
└── save_data()                   # Save to CSV files
```

---

### 2. `02_config.py`
**Purpose**: Setup configuration and directories  
**Run order**: 2nd  
**Execution time**: <1 second  
**Command**: `python 02_config.py`

**What it creates**:
```
feature_store/    # Engineered features
models/           # Trained models
logs/             # Execution logs
outputs/          # Forecasts
monitoring/       # Drift reports
metadata/         # Metadata files
reports/          # Generated reports
config.json       # Saved configuration
```

**When to modify**:
- Change hyperparameters
- Adjust thresholds
- Modify feature engineering settings

---

### 3. `03_pipeline_ingestion.py`
**Purpose**: Data ingestion and feature engineering  
**Run order**: 3rd  
**Execution time**: 1-2 minutes  
**Command**: `python 03_pipeline_ingestion.py`

**What it creates**:
```
feature_store/
├── features_v1.pkl       # Engineered features (55 features)
└── metadata_v1.json      # Feature metadata
```

**Class Structure**:
```python
DataValidator
├── validate_schema()        # Check required columns
├── detect_outliers()        # Find anomalies
└── check_data_quality()     # Comprehensive quality checks

FeatureEngineer
├── create_temporal_features()      # Time-based (day, month, etc.)
├── create_lag_features()           # Historical values (7d, 14d, 30d, 365d)
├── create_rolling_features()       # Statistics (mean, std, max, min)
├── create_hierarchical_features()  # Store/brand/country embeddings
└── create_all_features()           # Orchestrate all feature creation

DataIngestionPipeline
├── load_raw_data()         # Load CSV files
├── validate_data()         # Quality checks
├── process_and_engineer_features()  # Create features
└── save_to_feature_store() # Save to pickle file
```

**55 Features Created**:
```
Temporal (15):
  - day_of_week, day_of_month, month, quarter, year, week_of_year
  - is_weekend, month_sin, month_cos, day_of_week_sin, day_of_week_cos
  - days_to_black_friday, days_to_new_year

Lag (4):
  - sales_amount_lag_7d, sales_amount_lag_14d
  - sales_amount_lag_30d, sales_amount_lag_365d

Rolling (12): (mean, std, max, min for 7d, 14d, 30d)
  - sales_amount_rolling_mean_7d, sales_amount_rolling_std_7d, etc.

Hierarchical (10):
  - store_size_sqm, location_type_encoded, tier_encoded
  - country_encoded, category_encoded, price_tier_encoded
  - avg_price_point, store_cluster

External (5):
  - gdp_growth, tourism_index, currency_rate_to_usd
  - temperature_celsius, is_public_holiday

Promotion (2):
  - promotion_flag, promotion_discount

Transactions (2):
  - num_transactions, avg_transaction_value
```

---

### 4. `04_pipeline_training.py`
**Purpose**: Train global model and local adapters  
**Run order**: 4th  
**Execution time**: 5-10 minutes  
**Command**: `python 04_pipeline_training.py`

**What it creates**:
```
models/
├── global_model_v1.pkl          # Gradient boosting model
├── local_adapters_v1.pkl        # Store-specific adjustments
└── training_results_v1.json     # Performance metrics
```

**Class Structure**:
```python
DataSplitter
└── time_based_split()    # 80% train, 10% val, 10% test

GlobalModel
├── prepare_features()    # Select feature columns
├── train()               # Train gradient boosting
├── predict()             # Generate predictions
├── get_feature_importance()  # Top features
├── save()                # Save model to file
└── load()                # Load model from file

LocalAdapter
├── fit_store_adapters()           # Fit residual models
├── predict_with_adaptation()      # Ensemble prediction
├── save()                         # Save adapters
└── load()                         # Load adapters

ModelTrainingPipeline
├── load_features()       # Load from feature store
├── split_data()          # Train/val/test split
├── train_models()        # Train global + local
├── evaluate_models()     # Calculate metrics
└── save_models()         # Save to disk
```

**Key Algorithm**:
```python
# Global Model
Gradient Boosting with 200 trees, depth 7
Trained on ALL 500 stores combined

# Local Adaptation
For each store with 180+ days of data:
  1. Calculate residual = actual - global_prediction
  2. Store mean_residual and std_residual
  3. confidence = min(data_points / 1095, 1.0)

# Final Prediction
prediction = global_pred + (shrinkage × confidence × local_residual)
where shrinkage = 0.7 (configurable)
```

**Expected Performance**:
- Global Model: 6-8% MAPE
- Adapted Model: 5-7% MAPE
- Improvement: 1-2% MAPE reduction

---

### 5. `05_pipeline_inference.py`
**Purpose**: Generate forecasts for future dates  
**Run order**: 5th  
**Execution time**: 2-3 minutes  
**Command**: `python 05_pipeline_inference.py`

**What it creates**:
```
outputs/
├── forecasts_next_30days.csv    # Detailed daily forecasts
├── forecast_summary.csv         # Aggregated by store
└── forecast_report.txt          # Summary statistics
```

**Class Structure**:
```python
ForecastGenerator
├── create_future_dates()            # Generate date range
├── prepare_forecast_features()      # Create features for future
├── generate_forecasts()             # Make predictions
└── _days_to_holiday()               # Helper function

InferencePipeline
├── load_models()        # Load global + local models
├── load_data()          # Load stores, brands, historical
├── run()                # Orchestrate inference
├── save_forecasts()     # Save to CSV
└── generate_summary()   # Create report
```

**Forecast Output Format**:
```csv
date,store_id,brand_id,country,predicted_sales,lower_bound,upper_bound,confidence_level,prediction_type
2026-02-01,STORE_001,BRAND_15,UAE,2847.32,2278.86,3415.78,0.8,adapted
2026-02-01,STORE_002,BRAND_08,UAE,1523.45,1218.76,1828.14,0.8,global_only
...
```

**Key Features**:
- 30-day forecast horizon (configurable)
- 80% confidence intervals (lower/upper bounds)
- Prediction type flag (adapted vs global_only)
- Non-negative sales constraint

---

### 6. `06_pipeline_monitoring.py`
**Purpose**: Drift detection and model monitoring  
**Run order**: 6th  
**Execution time**: 1-2 minutes  
**Command**: `python 06_pipeline_monitoring.py`

**What it creates**:
```
monitoring/
├── drift_detection_report.json         # Detailed drift analysis
├── store_health_dashboard.csv          # MAPE by store
└── retraining_recommendations.txt      # Action items
```

**Class Structure**:
```python
MAPEMonitor
├── calculate_rolling_mape()   # Rolling 7-day MAPE
└── detect_mape_drift()        # Check thresholds

PSICalculator
└── calculate_psi()            # Population Stability Index
└── detect_feature_drift()     # Check each feature

CUSUMMonitor
├── update()                   # Update cumulative sum
└── reset()                    # Reset counters

DriftDetector
├── detect_all_drift()         # Run all 3 methods
└── _determine_action()        # Decide response

MonitoringPipeline
├── load_predictions_and_actuals()  # Load data
├── generate_predictions()          # Get model predictions
├── analyze_store_health()          # MAPE by store
├── save_results()                  # Save reports
└── generate_report()               # Create summary
```

**3 Drift Detection Methods**:

1. **MAPE-Based (Error Monitoring)**
   ```
   Minor:    5-8% MAPE    → Alert only
   Moderate: 8-12% MAPE   → Auto-retrain cluster
   Severe:   >12% MAPE    → Emergency global retrain
   ```

2. **PSI (Feature Distribution Shift)**
   ```
   PSI < 0.1:     No significant change
   0.1 < PSI < 0.25:  Small change, monitor
   PSI > 0.25:    Significant drift, take action
   ```

3. **CUSUM (Gradual Drift)**
   ```
   Cumulative sum of forecast errors
   Detects slow degradation over time
   ```

---

##  Special Scripts

### `RUN_ALL.py`
**Purpose**: Execute all 6 pipelines in order  
**Execution time**: 15-25 minutes total  
**Command**: `python RUN_ALL.py`

**What it does**:
1. Runs `01_generate_data.py`
2. Runs `02_config.py`
3. Runs `03_pipeline_ingestion.py`
4. Runs `04_pipeline_training.py`
5. Runs `05_pipeline_inference.py`
6. Runs `06_pipeline_monitoring.py`
7. Shows summary of results

**When to use**:
- First time setup
- Complete end-to-end run
- After making configuration changes

---

##  Documentation Files

### `README.md`
**Purpose**: Complete project documentation  
**Contents**:
- Overview and architecture
- Installation instructions
- Detailed module descriptions
- Configuration reference
- Performance metrics
- Business value analysis
- Troubleshooting guide

**Read this**: To understand the complete system



### `ARCHITECTURE.md`
**Purpose**: Technical architecture details  
**Contents**:
- ASCII architecture diagrams
- Pipeline flow visualization
- Algorithm details
- Retraining strategy
- Cold-start solutions
- Performance benchmarks

**Read this**: To understand technical design decisions

---

## Data Files

### Generated by `01_generate_data.py`:
```
data/stores.csv           # 508 rows, 9 columns
data/brands.csv           # 46 rows, 6 columns
data/sales.csv            # ~416,000 rows, 13 columns
data/external.csv         # ~8,768 rows, 6 columns
data/summary.json         # Dataset summary
```

### Generated by `03_pipeline_ingestion.py`:
```
feature_store/features_v1.pkl      # ~416,000 rows, 55 columns
feature_store/metadata_v1.json     # Feature information
```

### Generated by `04_pipeline_training.py`:
```
models/global_model_v1.pkl         # ~5-10 MB
models/local_adapters_v1.pkl       # ~100-500 KB
models/training_results_v1.json    # Performance metrics
```

### Generated by `05_pipeline_inference.py`:
```
outputs/forecasts_next_30days.csv  # ~15,000 rows (508 stores × 30 days)
outputs/forecast_summary.csv       # 508 rows (one per store)
outputs/forecast_report.txt        # Text summary
```

### Generated by `06_pipeline_monitoring.py`:
```
monitoring/drift_detection_report.json           # JSON with drift metrics
monitoring/store_health_dashboard.csv            # Store-level MAPE
monitoring/retraining_recommendations.txt        # Action recommendations
```

---

## Data Flow

```
01_generate_data.py
    ↓
data/*.csv
    ↓
03_pipeline_ingestion.py
    ↓
feature_store/features_v1.pkl
    ↓
04_pipeline_training.py
    ↓
models/*.pkl
    ↓
05_pipeline_inference.py → outputs/*.csv
    ↓
06_pipeline_monitoring.py → monitoring/*
```

---

##  Which File to Modify for What?

### To change model hyperparameters:
**Edit**: `config.py` (lines 40-60)  
**Then run**: `02_config.py`, then `04_pipeline_training.py`

### To change drift thresholds:
**Edit**: `config.py` (lines 70-90)  
**Then run**: `02_config.py`, then `06_pipeline_monitoring.py`

### To add new features:
**Edit**: `03_pipeline_ingestion.py` (FeatureEngineer class)  
**Then run**: `03_pipeline_ingestion.py`, then `04_pipeline_training.py`

### To change forecast horizon:
**Edit**: `05_pipeline_inference.py` (line in main: horizon_days=30)  
**Then run**: `05_pipeline_inference.py`

### To add new synthetic data patterns:
**Edit**: `01_generate_data.py` (generate_sales_data method)  
**Then run**: Full pipeline from step 1

---

##  File Size Reference

```
Small (<1 MB):
  - All .py files
  - config.json
  - All documentation files
  - All JSON outputs

Medium (1-50 MB):
  - data/sales.csv (~20-30 MB)
  - feature_store/features_v1.pkl (~40-50 MB)
  - outputs/forecasts_next_30days.csv (~1-2 MB)

Large (>50 MB):
  - None (this is a demo-scale implementation)
```

---

##  Imports Reference

Each script imports from `config.py`, so ensure all files are in the same directory:

```
sales-forecasting-platform/
├── 01_generate_data.py
├── 02_config.py
├── 03_pipeline_ingestion.py    ← imports config
├── 04_pipeline_training.py     ← imports config
├── 05_pipeline_inference.py    ← imports config, pipeline_2_training
├── 06_pipeline_monitoring.py   ← imports config, pipeline_2_training
├── RUN_ALL.py
├── config.py                   ← base module
├── QUICK_START.md
└── ARCHITECTURE.md
```

---
