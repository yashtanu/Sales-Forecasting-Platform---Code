"""
Configuration Module - Simple Import Version
This file contains only the configuration classes for importing
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: str = 'data'
    feature_store_dir: str = 'feature_store'
    model_dir: str = 'models'
    output_dir: str = 'outputs'
    
    stores_file: str = 'stores.csv'
    brands_file: str = 'brands.csv'
    sales_file: str = 'sales.csv'
    external_file: str = 'external.csv'
    
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ModelConfig:
    """Model training configuration"""
    model_type: str = 'gradient_boosting'
    lag_features: List[int] = None
    rolling_windows: List[int] = None
    
    data_rich_threshold_days: int = 1095
    data_medium_threshold_days: int = 365
    min_data_days: int = 180
    
    use_embeddings: bool = True
    embedding_dim: int = 10
    
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 7
    subsample: float = 0.8
    shrinkage_factor: float = 0.7
    
    def __post_init__(self):
        if self.lag_features is None:
            self.lag_features = [7, 14, 30, 365]
        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 30]


@dataclass
class DriftConfig:
    """Drift detection configuration"""
    mape_minor_threshold: float = 5.0
    mape_alert_threshold: float = 8.0
    mape_emergency_threshold: float = 12.0
    mape_increase_threshold: float = 0.3
    
    psi_threshold: float = 0.25
    cusum_threshold: float = 5.0
    cusum_drift_threshold: float = 3.0
    rolling_window_days: int = 7
    
    actions: Dict[str, str] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = {
                'minor': 'alert_only',
                'moderate': 'retrain_cluster',
                'severe': 'retrain_global_emergency'
            }


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 1000
    confidence_level: float = 0.8
    
    global_weight: float = 0.7
    local_weight: float = 0.3
    
    target_latency_ms: float = 50
    target_throughput_rps: int = 10000
    target_availability: float = 0.999


@dataclass
class PlatformConfig:
    """Overall platform configuration"""
    data: DataConfig
    model: ModelConfig
    drift: DriftConfig
    inference: InferenceConfig
    
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    random_seed: int = 42
    enable_monitoring: bool = True
    monitoring_interval_hours: int = 1
    
    @classmethod
    def create_default(cls):
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            drift=DriftConfig(),
            inference=InferenceConfig()
        )
    
    def to_dict(self):
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'drift': asdict(self.drift),
            'inference': asdict(self.inference),
            'log_level': self.log_level,
            'log_dir': self.log_dir,
            'random_seed': self.random_seed,
            'enable_monitoring': self.enable_monitoring,
            'monitoring_interval_hours': self.monitoring_interval_hours
        }


def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'feature_store', 'models', 'logs', 'outputs', 'monitoring', 'metadata', 'reports']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True, parents=True)
