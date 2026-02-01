#  Architecture Overview

## Sales Forecasting Platform - Technical Architecture

---

## System Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                   │
├───────────────────────────────────────────────────────────────────────┤
│  POS Systems  │  ERP/SAP  │  E-commerce  │  Marketing  │  External APIs│
└────────┬──────────────────┬────────────────┬───────────────┬───────────┘
         │                  │                │               │
         └──────────────────┴────────────────┴───────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    PIPELINE 1: DATA INGESTION                          │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐              │
│  │   Data      │  │   Feature    │  │   Feature       │              │
│  │ Validation  │→ │ Engineering  │→ │   Store         │              │
│  └─────────────┘  └──────────────┘  └─────────────────┘              │
│                                                                         │
│  Features Created:                                                     │
│  • Temporal (15): day_of_week, month, seasonality                     │
│  • Lag (4): 7d, 14d, 30d, 365d historical sales                       │
│  • Rolling (12): mean, std, max, min over 7/14/30d windows            │
│  • Hierarchical (10): store/brand/country embeddings                  │
│  • External (5): GDP, tourism, weather, currency                      │
│                                                                         │
│  Output: 55 features per record → Feature Store                       │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    PIPELINE 2: MODEL TRAINING                          │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │         TIER 1: GLOBAL FOUNDATION MODEL                  │         │
│  │                                                            │         │
│  │  Algorithm: Gradient Boosting                             │         │
│  │  Training Data: ALL 500 stores combined                   │         │
│  │  Purpose: Learn universal sales patterns                  │         │
│  │                                                            │         │
│  │  Configuration:                                            │         │
│  │    • n_estimators: 200                                    │         │
│  │    • max_depth: 7                                         │         │
│  │    • learning_rate: 0.05                                  │         │
│  │    • Features: All 55 engineered features                 │         │
│  └──────────────────────────────────────────────────────────┘         │
│                           │                                             │
│                           ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │      TIER 2: HIERARCHICAL FEATURES                        │         │
│  │                                                            │         │
│  │  • Country embeddings (8 countries)                       │         │
│  │  • Brand embeddings (46 brands)                           │         │
│  │  • Store cluster features (10 clusters)                   │         │
│  │  • Location type encoding (mall/street/airport)           │         │
│  └──────────────────────────────────────────────────────────┘         │
│                           │                                             │
│                           ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │      TIER 3: LOCAL ADAPTATION                             │         │
│  │                                                            │         │
│  │  Store-Specific Residual Models:                          │         │
│  │    • Only for stores with 180+ days of data               │         │
│  │    • Learn store-specific adjustment patterns             │         │
│  │    • Bayesian shrinkage to global baseline                │         │
│  │                                                            │         │
│  │  Prediction Formula:                                       │         │
│  │  prediction = global_pred +                                │         │
│  │              (shrinkage × confidence × local_residual)    │         │
│  │                                                            │         │
│  │  where:                                                    │         │
│  │    shrinkage = 0.7 (configurable)                        │         │
│  │    confidence = min(data_points / 1095, 1.0)             │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│  Performance Targets:                                                  │
│  • Data-rich stores: MAPE ≤ 5%                                        │
│  • Data-poor stores: MAPE ≤ 8%                                        │
│  • Forecast bias: ±2%                                                 │
│                                                                         │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    PIPELINE 3: INFERENCE                               │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐              │
│  │   Load      │  │   Prepare    │  │   Generate      │              │
│  │  Models     │→ │   Features   │→ │  Predictions    │              │
│  └─────────────┘  └──────────────┘  └─────────────────┘              │
│                                              │                         │
│                                              ▼                         │
│                                    ┌─────────────────┐                │
│                                    │   Prediction    │                │
│                                    │   Intervals     │                │
│                                    └─────────────────┘                │
│                                                                         │
│  Output Format:                                                        │
│  • Date, Store ID, Predicted Sales                                    │
│  • Lower Bound (80% CI)                                               │
│  • Upper Bound (80% CI)                                               │
│  • Prediction Type (global_only vs adapted)                           │
│                                                                         │
│  Performance:                                                          │
│  • Latency: <50ms (p95)                                               │
│  • Throughput: 10K requests/sec                                       │
│  • Availability: 99.9%                                                │
│                                                                         │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│              PIPELINE 4: DRIFT DETECTION & MONITORING                  │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Multi-Layered Drift Detection:                                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │ METHOD 1: Error-Based (MAPE)                             │         │
│  │                                                            │         │
│  │  • Track rolling 7-day MAPE by store                     │         │
│  │  • Alert if MAPE > 8% or increases >30%                  │         │
│  │  • Compare to baseline performance                       │         │
│  │                                                            │         │
│  │  Thresholds:                                              │         │
│  │    Minor:    5-8%    → Alert only                        │         │
│  │    Moderate: 8-12%   → Auto-retrain cluster              │         │
│  │    Severe:   >12%    → Emergency global retrain          │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │ METHOD 2: Statistical Tests (PSI)                        │         │
│  │                                                            │         │
│  │  • Population Stability Index on features                │         │
│  │  • Detects feature distribution shifts                   │         │
│  │                                                            │         │
│  │  PSI Interpretation:                                      │         │
│  │    <0.1:     No significant change                        │         │
│  │    0.1-0.25: Small change, monitor                       │         │
│  │    >0.25:    Significant drift, take action              │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │ METHOD 3: Residual Patterns (CUSUM)                      │         │
│  │                                                            │         │
│  │  • Cumulative sum of forecast errors                     │         │
│  │  • Detects gradual drift over time                       │         │
│  │  • Control limits for early warning                      │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │ METHOD 4: Business Context                               │         │
│  │                                                            │         │
│  │  Manual triggers:                                         │         │
│  │    • Store renovation                                     │         │
│  │    • New competitor                                       │         │
│  │    • Brand repositioning                                  │         │
│  │                                                            │         │
│  │  Economic events:                                         │         │
│  │    • Policy changes                                       │         │
│  │    • Currency fluctuation                                 │         │
│  │    • Tourism impact                                       │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│  Automated Decision Engine:                                            │
│  ┌──────────────────────┬──────────────────┬──────────────┐          │
│  │ Drift Level          │ Action           │ Scope        │          │
│  ├──────────────────────┼──────────────────┼──────────────┤          │
│  │ Minor (5-8%)         │ Alert only       │ Review       │          │
│  │ Moderate (8-12%)     │ Auto-retrain     │ 10-50 stores │          │
│  │ Severe (>12%)        │ Emergency        │ Global +     │          │
│  │                      │ retrain          │ human review │          │
│  └──────────────────────┴──────────────────┴──────────────┘          │
│                                                                         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Retraining Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                  RETRAINING SCHEDULE                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Global Model Retrain:                                       │
│  ├─ Frequency: Monthly                                       │
│  ├─ Trigger: Scheduled                                       │
│  ├─ Scope: All 500 stores                                   │
│  ├─ Cost: High (full retrain)                               │
│  └─ Purpose: Capture new patterns across portfolio          │
│                                                               │
│  Cluster Retrain:                                            │
│  ├─ Frequency: Weekly                                        │
│  ├─ Trigger: Moderate drift detected                        │
│  ├─ Scope: 10-50 stores in affected cluster                │
│  ├─ Cost: Medium                                             │
│  └─ Purpose: Update cluster-specific adjustments            │
│                                                               │
│  Store-Specific Tune:                                        │
│  ├─ Frequency: Daily                                         │
│  ├─ Trigger: Minor drift or new data                        │
│  ├─ Scope: 1-5 stores                                       │
│  ├─ Cost: Low (local adjustment only)                       │
│  └─ Purpose: Quick corrections for individual stores        │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Cost Savings:
  Baseline: Training 500 models daily = ~40 hours compute
  Our approach: 1 global monthly + targeted clusters = <2 hours
  Reduction: 95% cost savings
```

---

## Cold-Start Strategies

### Scenario 1: New Store Opening (Zero History)

```
┌─────────────────────────────────────────────────────────────┐
│  MONTH 0-3: Pure Cluster-Based Forecasting                  │
├─────────────────────────────────────────────────────────────┤
│  1. Cluster Assignment                                       │
│     └─ Match by: location tier, size, demographics          │
│                                                               │
│  2. Transfer Learning                                        │
│     └─ Use: Global model + cluster embeddings               │
│                                                               │
│  3. Expected MAPE: 15-20%                                    │
│     └─ Acceptable given zero history                        │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  MONTH 3-6: Blended Approach                                 │
├─────────────────────────────────────────────────────────────┤
│  • 70% cluster baseline                                      │
│  • 30% store-specific learning                              │
│  • Expected MAPE: 10-12%                                     │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  MONTH 6+: Full Local Adaptation                             │
├─────────────────────────────────────────────────────────────┤
│  • Bayesian shrinkage with increasing confidence             │
│  • Expected MAPE: 7-8%                                       │
│  • Approaching mature store accuracy                         │
└─────────────────────────────────────────────────────────────┘
```

### Scenario 2: New Brand (Limited Data)

```
┌─────────────────────────────────────────────────────────────┐
│  HIERARCHICAL POOLING APPROACH                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Level 1: Brand Baseline                                     │
│  └─ Use brand data from other countries                     │
│     Example: Brand has 15 stores in UAE with 3yr history    │
│                                                               │
│  Level 2: Country Adjustment                                 │
│  └─ Apply local macro indicators, preferences, pricing      │
│     Example: Bahrain market is smaller, different demos     │
│                                                               │
│  Level 3: Store Factor                                       │
│  └─ Use 6 months to fine-tune                               │
│                                                               │
│  Formula:                                                     │
│  Forecast = Brand_Baseline × Country_Factor × Store_Factor  │
│                                                               │
│  Heavy shrinkage toward brand mean when data is sparse       │
└─────────────────────────────────────────────────────────────┘
```

---

##  Business Value Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│              ESTIMATED ANNUAL BUSINESS VALUE                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Working Capital Optimization:                               │
│  ├─ 15% reduction in excess inventory                       │
│  ├─ Assuming $100M inventory                                │
│  └─ Value: $750K annual savings                             │
│                                                               │
│  Stockout Prevention:                                        │
│  ├─ 20% reduction in stockouts                              │
│  ├─ Luxury goods: 60%+ gross margin                         │
│  └─ Value: $2-3M recovered sales                            │
│                                                               │
│  Markdown Reduction:                                         │
│  ├─ 10% reduction in end-of-season markdowns                │
│  ├─ Better demand forecasting = less excess                 │
│  └─ Value: $1-2M margin protection                          │
│                                                               │
│  Operational Efficiency:                                     │
│  ├─ 90% reduction in manual forecasting                     │
│  ├─ Planning team focuses on strategy                       │
│  └─ Value: $200K labor reallocation                         │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│  TOTAL ANNUAL VALUE: $4-6M                                   │
│  ROI: 15-20×                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Targets vs Actuals

```
┌────────────────────────────┬──────────────┬──────────────┐
│ Metric                     │ Target       │ Expected     │
├────────────────────────────┼──────────────┼──────────────┤
│ Model Accuracy             │              │              │
│  • Data-rich stores        │ MAPE ≤ 5%    │ MAPE ~5-6%  │
│  • Data-poor stores        │ MAPE ≤ 8%    │ MAPE ~6-8%  │
│  • Forecast bias           │ ±2%          │ ±1-2%        │
│                             │              │              │
│ Operational Metrics        │              │              │
│  • Maintenance time        │ <4 hrs/month │ ~3 hrs/month│
│  • Drift detection         │ 100% stores  │ 100% stores │
│  • Forecast latency        │ <50ms (p95)  │ ~30-40ms    │
│  • Availability            │ 99.9%        │ 99.9%+       │
│                             │              │              │
│ Business Impact            │              │              │
│  • Inventory reduction     │ 15%          │ 15-20%       │
│  • Stockout reduction      │ 20%          │ 20-25%       │
│  • Markdown reduction      │ 10%          │ 10-15%       │
└────────────────────────────┴──────────────┴──────────────┘
```

---

##  Key Technology Choices

**Why Gradient Boosting (not Neural Networks)?**
- Works well with tabular data
- Handles missing values naturally
- Interpretable (feature importance)
- Fast inference (<50ms)
- Less data hungry than deep learning

**Why Hierarchical Structure?**
- Enables transfer learning across stores
- Handles data sparsity elegantly
- Single model easier to maintain than 500
- Automatic knowledge sharing

**Why Bayesian Shrinkage?**
- Balances global vs local patterns
- Adapts based on data confidence
- Prevents overfitting on new stores
- Graceful degradation for cold-start

---

## Scaling Considerations

**Current Setup (Demo)**
- 500 stores, ~416K records
- Training time: ~5-10 minutes
- Inference: ~2-3 minutes

**Production Scale**
- Could handle 5,000+ stores
- Distributed training on cloud
- Real-time inference with caching
- Automated retraining pipelines

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
