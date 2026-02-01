"""
Configuration Module for Sales Forecasting Platform
Centralized configuration for all pipelines
"""

import os
import json
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
    
    # File paths
    stores_file: str = 'stores.csv'
    brands_file: str = 'brands.csv'
    sales_file: str = 'sales.csv'
    external_file: str = 'external.csv'
    
    # Data split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ModelConfig:
    """Model training configuration"""
    # Model type
    model_type: str = 'gradient_boosting'
    
    # Feature engineering
    lag_features: List[int] = None
    rolling_windows: List[int] = None
    
    # Cold-start thresholds (in days)
    data_rich_threshold_days: int = 1095  # 3 years
    data_medium_threshold_days: int = 365  # 1 year
    min_data_days: int = 180  # 6 months minimum
    
    # Hierarchical settings
    use_embeddings: bool = True
    embedding_dim: int = 10
    
    # Model hyperparameters
    n_estimators: int = 200  # Reduced for demo speed
    learning_rate: float = 0.05
    max_depth: int = 7
    subsample: float = 0.8
    
    # Bayesian shrinkage
    shrinkage_factor: float = 0.7  # How much to trust local vs global
    
    def __post_init__(self):
        if self.lag_features is None:
            # Lag features: 1 week, 2 weeks, 1 month, 1 year
            self.lag_features = [7, 14, 30, 365]
        
        if self.rolling_windows is None:
            # Rolling statistics windows
            self.rolling_windows = [7, 14, 30]


@dataclass
class DriftConfig:
    """Drift detection configuration"""
    # Error-based monitoring
    mape_minor_threshold: float = 5.0  # Below this is good
    mape_alert_threshold: float = 8.0  # Alert if MAPE > 8%
    mape_emergency_threshold: float = 12.0  # Emergency if MAPE > 12%
    mape_increase_threshold: float = 0.3  # Alert if MAPE increases by 30%
    
    # Statistical tests
    psi_threshold: float = 0.25  # Population Stability Index threshold
    
    # CUSUM parameters for gradual drift
    cusum_threshold: float = 5.0
    cusum_drift_threshold: float = 3.0
    
    # Monitoring windows
    rolling_window_days: int = 7
    
    # Retraining decision rules
    actions: Dict[str, str] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = {
                'minor': 'alert_only',  # MAPE 5-8%
                'moderate': 'retrain_cluster',  # MAPE 8-12%
                'severe': 'retrain_global_emergency'  # MAPE > 12%
            }


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 1000
    confidence_level: float = 0.8  # For prediction intervals
    
    # Ensemble weights
    global_weight: float = 0.7
    local_weight: float = 0.3
    
    # Performance targets (from presentation)
    target_latency_ms: float = 50
    target_throughput_rps: int = 10000
    target_availability: float = 0.999  # 99.9% SLA


@dataclass
class PlatformConfig:
    """Overall platform configuration"""
    data: DataConfig
    model: ModelConfig
    drift: DriftConfig
    inference: InferenceConfig
    
    # System settings
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    random_seed: int = 42
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval_hours: int = 1
    
    @classmethod
    def create_default(cls):
        """Create default configuration"""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            drift=DriftConfig(),
            inference=InferenceConfig()
        )
    
    def to_dict(self):
        """Convert to dictionary"""
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
    
    def save(self, filepath: str = 'config.json'):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = 'config.json'):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            drift=DriftConfig(**config_dict['drift']),
            inference=InferenceConfig(**config_dict['inference'])
        )


def setup_directories():
    """Create necessary directories for the platform"""
    directories = [
        'data',
        'feature_store',
        'models',
        'logs',
        'outputs',
        'monitoring',
        'metadata',
        'reports'
    ]
    
    print("Setting up directory structure...")
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True, parents=True)
        print(f"  ✓ {dir_name}/")
    
    print("Directory structure created!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" CONFIGURATION SETUP")
    print("="*70)
    
    # Create directories
    setup_directories()
    
    # Create and save default configuration
    print("\nCreating default configuration...")
    config = PlatformConfig.create_default()
    config.save('config.json')
    
    print("\n" + "="*70)
    print(" CONFIGURATION DETAILS")
    print("="*70)
    print(f"\nModel Configuration:")
    print(f"  • Model Type: {config.model.model_type}")
    print(f"  • N Estimators: {config.model.n_estimators}")
    print(f"  • Learning Rate: {config.model.learning_rate}")
    print(f"  • Max Depth: {config.model.max_depth}")
    
    print(f"\nDrift Detection:")
    print(f"  • MAPE Alert Threshold: {config.drift.mape_alert_threshold}%")
    print(f"  • MAPE Emergency Threshold: {config.drift.mape_emergency_threshold}%")
    print(f"  • PSI Threshold: {config.drift.psi_threshold}")
    
    print(f"\nCold-Start Thresholds:")
    print(f"  • Data Rich: {config.model.data_rich_threshold_days} days")
    print(f"  • Data Medium: {config.model.data_medium_threshold_days} days")
    print(f"  • Minimum Data: {config.model.min_data_days} days")
    
    print("\n" + "="*70)
    print(" Configuration setup complete!")
    print("="*70)
    print("\nNext: Run the data generation script (01_generate_data.py)")
    print("\n")
