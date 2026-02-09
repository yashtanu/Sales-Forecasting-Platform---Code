# Cloud-Native Managed Services Evaluation

## Context: Sales Forecasting Platform

This document evaluates how managed cloud-native services (e.g., AWS SageMaker) could be incorporated into this platform's architecture, and the decision factors for choosing them over traditional EC2-based deployments. The analysis is grounded in the actual current state of this codebase: a local, single-machine batch ML pipeline using scikit-learn Gradient Boosting, file-based persistence (CSV/pickle), and manual orchestration via `RUN_ALL.py`.

---

## 1. Current Architecture and Its Limitations

The platform today runs entirely on a single machine:

| Component | Current Implementation | Production Gap |
|---|---|---|
| Training | `04_pipeline_training.py` — scikit-learn GradientBoosting on local CPU | No distributed training, no GPU, no hyperparameter search at scale |
| Feature Store | Pickle files in `/feature_store/` | No versioning, no serving layer, no shared access |
| Model Registry | `models/global_model_v1.pkl` + JSON metadata | Single version, no lineage, no rollback capability |
| Inference | `05_pipeline_inference.py` — batch predictions to CSV | No real-time API, no autoscaling, no A/B testing |
| Monitoring | `06_pipeline_monitoring.py` — MAPE/PSI/CUSUM to JSON | Post-hoc only, no alerting, no automated retraining triggers |
| Orchestration | `RUN_ALL.py` calling `subprocess.run()` sequentially | No scheduling, no retry logic, no DAG dependencies |
| Data Storage | CSV files in `/data/` (~416K records) | No query engine, no partitioning, no concurrent access |

This setup works well for prototyping and proof-of-concept with 508 stores and ~416K records. The question is what changes when this needs to serve production traffic, scale to more stores, or support organizational ML operations.

---

## 2. Where Managed Services Map to This Platform

### 2.1 Training: SageMaker Training Jobs vs. EC2

**Current state:** The global model trains on all 500 stores combined using `GradientBoostingRegressor` with 200 estimators. Local adaptation fits residual models for stores with 180+ days of data. This runs in minutes on a single machine.

**When EC2 is sufficient:**
- The current dataset (~416K rows, 55 features) fits comfortably in memory on a modest instance. A `c5.2xlarge` running the existing scikit-learn code would handle this without modification.
- If the team already manages EC2 infrastructure and has AMIs with the right Python environment, there is no strong reason to introduce SageMaker for a workload this size.
- EC2 gives full control over the runtime environment, which matters when debugging training failures or profiling performance.

**When SageMaker Training Jobs become the better choice:**
- **Data growth.** If the platform expands from 508 stores to thousands, or ingests transaction-level data instead of daily aggregates, the dataset could grow 10-100x. SageMaker's managed training handles data sharding, instance provisioning, and teardown automatically.
- **Hyperparameter optimization.** The current config hardcodes `n_estimators=200`, `max_depth=7`, `learning_rate=0.05`. SageMaker Automatic Model Tuning runs parallel Bayesian optimization across instances, which is difficult to replicate on a single EC2 box.
- **Algorithm experimentation.** If the team wants to evaluate XGBoost, LightGBM, or neural approaches (e.g., temporal fusion transformers), SageMaker's built-in algorithm containers and framework support reduce setup friction compared to managing EC2 environments per framework.
- **Spot instance management.** SageMaker Training Jobs handle spot interruptions with checkpointing. On raw EC2, the team would need to build this themselves.
- **Cost.** Training jobs spin up and terminate automatically. With EC2, forgotten running instances are a common source of waste. For periodic retraining (the monitoring pipeline recommends cluster-level and global retraining), pay-per-job is more cost-efficient than a standing instance.

### 2.2 Feature Store: SageMaker Feature Store vs. Self-Managed

**Current state:** `03_pipeline_ingestion.py` engineers 55 features and writes them to `feature_store/features_v1.pkl` (~170 MB) with a JSON metadata sidecar.

**When self-managed is sufficient:**
- A single pipeline reads and writes features. There is no concurrent access pattern, no feature sharing across teams, and no online serving requirement. Pickle files work.

**When a managed Feature Store earns its overhead:**
- **Online + offline serving.** If the platform adds a real-time forecasting API (the architecture doc targets <50ms p95 latency), features need to be served from a low-latency store, not deserialized from pickle on each request. SageMaker Feature Store provides both an offline store (S3/Parquet for training) and an online store (single-digit-ms reads for inference).
- **Feature consistency.** The current system computes features at training time and again at inference time (`05_pipeline_inference.py` re-engineers features). A Feature Store ensures the same feature values are used in both paths, eliminating training-serving skew.
- **Feature reuse across models.** If the team builds additional models (e.g., demand planning, inventory optimization), a centralized feature store avoids redundant computation of lag features, rolling statistics, and hierarchical embeddings.

### 2.3 Model Hosting: SageMaker Endpoints vs. EC2 + Flask/FastAPI

**Current state:** No serving layer exists. `05_pipeline_inference.py` generates batch forecasts to CSV.

**When EC2 hosting is the right call:**
- A simple FastAPI server on EC2 loading the ~3 MB global model pickle would achieve the <50ms latency target easily. For a single model behind an ALB, this is straightforward to reason about, deploy, and debug.
- If the team has existing EC2 operational practices (monitoring, patching, deployment scripts), staying on EC2 avoids introducing a new abstraction layer.

**When SageMaker Endpoints become preferable:**
- **Autoscaling.** The platform serves 508 stores across 8 countries in different time zones. Traffic patterns will be uneven. SageMaker endpoints auto-scale on invocations per instance, CPU utilization, or custom metrics. On EC2, the team would need to configure ASGs, target tracking policies, and health checks manually.
- **Multi-model endpoints.** The architecture uses a global model plus per-store local adapters. SageMaker Multi-Model Endpoints can host thousands of models behind a single endpoint, loading them on demand. This maps directly to the global + local adapter architecture without managing a custom model routing layer.
- **A/B testing and shadow deployments.** When retraining produces a new model version, SageMaker production variants allow traffic splitting (e.g., 90/10) to validate the new model before full rollout. On EC2, this requires custom load-balancer configuration.
- **Model monitoring integration.** SageMaker Model Monitor can continuously evaluate data quality and model quality against a baseline, feeding directly into the drift detection system that `06_pipeline_monitoring.py` implements manually today.

### 2.4 Orchestration: Step Functions / MWAA vs. EC2 Cron

**Current state:** `RUN_ALL.py` executes 6 scripts sequentially via `subprocess.run()`. No retry logic, no conditional execution, no parallelism.

**When a cron job on EC2 works:**
- For a daily batch run that takes minutes and rarely fails, cron is simple and observable. Add basic error handling (exit codes, email on failure) and it covers the need.

**When managed orchestration is justified:**
- **Conditional retraining.** The monitoring pipeline outputs severity levels (minor/moderate/severe) and retraining recommendations. An orchestrator like Step Functions or MWAA (Managed Airflow) can trigger different retraining paths based on drift severity — cluster-level retrain for moderate drift, full global retrain for severe drift — without manual intervention.
- **Parallelism.** Feature engineering, store clustering, and external data fetching are independent and could run in parallel. The current sequential execution leaves performance on the table. Airflow DAGs or Step Functions parallel states express this naturally.
- **Observability.** Managed orchestrators provide execution history, duration metrics, failure traces, and retry logs. This is the first thing teams need when debugging "why did last night's forecast run fail?"

### 2.5 Data Storage: S3 + Athena/Redshift vs. Local CSV

**Current state:** Raw data in CSV files, ~416K records in `/data/`.

**When local files work:**
- The entire dataset fits in memory. Pandas reads it in seconds. There is no multi-user access pattern.

**When managed storage becomes necessary:**
- **Data volume.** Transaction-level data for 500+ stores will quickly exceed what's practical to store and query as flat CSV files.
- **Data lake pattern.** S3 as the storage layer with Athena for ad-hoc queries and Glue for ETL creates a scalable foundation. The current CSV → pandas pattern doesn't change much conceptually (pandas can read from S3 directly), but the storage layer becomes durable, versioned, and accessible to other tools.
- **Compliance and access control.** S3 bucket policies and IAM roles provide audit trails and access restrictions that local file permissions cannot match.

---

## 3. Decision Framework

The choice between managed services and EC2-based setups is not binary. It depends on concrete factors:

### Choose EC2 (or equivalent self-managed compute) when:

1. **The workload is well-understood and stable.** The current Gradient Boosting training job is predictable in resource consumption. No need for managed infrastructure to handle variability.
2. **The team has strong infrastructure expertise.** If the team can manage instances, networking, patching, and monitoring effectively, the additional abstraction of managed services adds cost without proportional benefit.
3. **Cost sensitivity at small scale.** A single `c5.xlarge` running 24/7 costs ~$125/month. SageMaker training jobs for equivalent compute cost more per hour but charge only for training duration. At low utilization (a few training runs per month), the on-demand EC2 or even a reserved instance may be cheaper.
4. **Debugging and customization matter most.** Full SSH access, custom profiling tools, arbitrary library installation — EC2 gives unrestricted control that managed services constrain.

### Choose managed services (SageMaker, Step Functions, etc.) when:

1. **The ML lifecycle needs automation.** This platform's monitoring pipeline already identifies when retraining is needed. Connecting that signal to automated retraining, model evaluation, and deployment is the natural next step — and that is exactly what SageMaker Pipelines orchestrates.
2. **Multiple models and versions need to coexist.** The current system stores only `v1`. Production use requires versioning, comparison, rollback, and gradual rollout. SageMaker Model Registry handles this out of the box.
3. **The team wants to focus on ML, not infrastructure.** Managed services trade control for operational simplicity. If the team is primarily data scientists (not DevOps engineers), SageMaker lets them train, deploy, and monitor models without managing instances, containers, or networking.
4. **Scale is variable or growing.** SageMaker endpoints autoscale. EC2 autoscaling is possible but requires more configuration. For a platform that could expand from 508 to 5,000+ stores, elastic infrastructure matters.
5. **Compliance and governance are requirements.** SageMaker provides model lineage, experiment tracking, bias detection, and audit logs natively. Building these on EC2 means integrating MLflow, Weights & Biases, or custom solutions.

---

## 4. Recommended Migration Path for This Platform

Given the current state of the codebase (prototype-quality, local execution, no cloud infrastructure), a staged approach makes sense:

### Stage 1: Lift and shift to EC2 + S3
- Move data to S3 (replace local CSV I/O with `s3://` paths in pandas)
- Run existing scripts on an EC2 instance via cron or SSM Run Command
- Store model artifacts in S3 instead of local `/models/`
- **Why EC2 first:** The code changes are minimal. The team gets cloud durability and access control without rewriting pipelines.

### Stage 2: Add orchestration and serving
- Replace `RUN_ALL.py` with Step Functions or Airflow (MWAA) DAG
- Deploy a FastAPI inference endpoint on ECS/Fargate or a SageMaker endpoint
- Connect the monitoring pipeline to CloudWatch alarms for drift alerting
- **Why not SageMaker yet:** The model is scikit-learn, the dataset is moderate, and the training logic is straightforward. Orchestration and serving are the higher-impact improvements.

### Stage 3: Adopt SageMaker for ML lifecycle
- Migrate training to SageMaker Training Jobs (enables hyperparameter tuning, spot training)
- Use SageMaker Feature Store to unify online/offline feature access
- Use SageMaker Model Registry for versioning and approval workflows
- Use SageMaker Model Monitor to replace custom `06_pipeline_monitoring.py` drift detection
- **Why now:** At this stage, the operational overhead of managing the full ML lifecycle on raw EC2 exceeds the overhead of learning SageMaker's abstractions.

---

## 5. Cost Comparison (Illustrative)

For a workload of this platform's current scale (508 stores, daily retraining, batch + API inference):

| Component | EC2-Based Monthly Cost | SageMaker-Based Monthly Cost |
|---|---|---|
| Training (daily, ~10 min) | $125 (c5.xlarge reserved) | $15 (ml.m5.xlarge × 0.33 hrs × 30 × $0.23/hr spot) |
| Inference endpoint | $150 (c5.xlarge + ALB) | $175 (ml.m5.large endpoint 24/7) |
| Feature Store | $5 (S3 storage) | $30 (online + offline store) |
| Orchestration | $0 (cron) | $50 (Step Functions / MWAA) |
| Monitoring | $0 (custom scripts) | $20 (Model Monitor) |
| **Total** | **~$280/month** | **~$290/month** |

At this scale, costs are comparable. The managed approach pays off in reduced operational burden and faster iteration, not raw compute savings. At larger scale (thousands of stores, multiple models, real-time serving), managed services typically become cheaper because they eliminate idle resources.

---

## 6. Summary

For this Sales Forecasting Platform specifically:

- **Today**, the local batch pipeline is appropriate for its current role as a prototype serving 508 stores with a single model architecture.
- **EC2 is the right first step** for moving to production. The code requires minimal changes, and the team gains cloud durability without learning new ML platform abstractions.
- **SageMaker and managed services become the right choice** when the platform needs automated retraining triggered by drift detection, model versioning with approval gates, real-time serving with autoscaling, or hyperparameter optimization across algorithm families.
- **The deciding factor is operational complexity, not compute.** The question is not whether SageMaker can train a Gradient Boosting model faster than EC2 (it cannot — the compute is the same). The question is whether the team wants to build and maintain the surrounding lifecycle infrastructure (versioning, monitoring, deployment, scaling) themselves or use a managed platform that provides it.

The three-tier training architecture (global model → hierarchical embeddings → local adaptation) maps well to SageMaker's abstractions: a single Training Job for the global model, a Processing Job for store clustering and feature engineering, and Multi-Model Endpoints for serving the global model plus per-store adapters. This alignment makes eventual migration straightforward without architectural redesign.
