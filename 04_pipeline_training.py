"""
Pipeline 2: Model Training
Implements global model + hierarchical local adaptation architecture
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
try:
    from config import ModelConfig, DataConfig
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "02_config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    ModelConfig = config_module.ModelConfig
    DataConfig = config_module.DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Time-based data splitting for forecasting"""
    
    @staticmethod
    def time_based_split(df: pd.DataFrame, 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by time"""
        df = df.sort_values(date_col)
        n = len(df)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        logger.info(f"Data split: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
        
        return train, val, test


class StoreClusterer:
    """Cluster stores for cold-start and transfer learning"""
    
    def __init__(self):
        from sklearn.cluster import KMeans
        self.clusterer = KMeans(n_clusters=10, random_state=42)
        self.cluster_labels = None
    
    def fit(self, stores_df: pd.DataFrame) -> 'StoreClusterer':
        """Cluster stores based on metadata"""
        # Features for clustering
        from sklearn.preprocessing import LabelEncoder
        
        features = stores_df.copy()
        
        # Encode categoricals
        le_location = LabelEncoder()
        features['location_encoded'] = le_location.fit_transform(features['location_type'])
        
        le_country = LabelEncoder()
        features['country_encoded'] = le_country.fit_transform(features['country'])
        
        # Clustering features
        X = features[['store_size_sqm', 'location_encoded', 'country_encoded', 'history_days']]
        X = X.fillna(0)
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        self.cluster_labels = self.clusterer.fit_predict(X_scaled)
        features['cluster'] = self.cluster_labels
        
        logger.info(f"Stores clustered into {len(np.unique(self.cluster_labels))} groups")
        
        return self
    
    def get_cluster_stats(self, stores_df: pd.DataFrame) -> Dict:
        """Get statistics per cluster"""
        stores_df['cluster'] = self.cluster_labels
        stats = {}
        
        for cluster_id in range(self.clusterer.n_clusters):
            cluster_stores = stores_df[stores_df['cluster'] == cluster_id]
            stats[cluster_id] = {
                'num_stores': len(cluster_stores),
                'avg_size': cluster_stores['store_size_sqm'].mean(),
                'avg_history_days': cluster_stores['history_days'].mean(),
                'countries': cluster_stores['country'].value_counts().to_dict()
            }
        
        return stats


class GlobalModel:
    """Global forecasting model trained on all stores"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_cols = None
        self.target_col = 'sales_amount'
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        # Exclude non-feature columns
        exclude_cols = [
            'date', 'store_id', 'brand_id', 'store_name',
            self.target_col, 'brand_name', 'country',
            'location_type', 'tier', 'category', 'price_tier',
            'opening_date', 'is_ecommerce', 'store_cluster'
        ]
        
        self.feature_cols = [col for col in df.columns 
                            if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Handle NaN
        X = X.fillna(0)
        
        logger.info(f"Features prepared: {len(self.feature_cols)} features")
        
        return X, y
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> 'GlobalModel':
        """Train global Gradient Boosting model"""
        logger.info("Training global model (this may take a few minutes)...")
        
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        
        # Initialize model with reduced parameters for faster training
        self.model = GradientBoostingRegressor(
            n_estimators=200,  # Reduced for demo
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=0.8,
            random_state=42,
            verbose=1
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate - filter out zero/near-zero sales for accurate MAPE
        y_pred = self.predict(val_df)
        
        # Only calculate MAPE on non-zero sales
        valid_mask = y_val > 1.0  # Filter out sales less than $1
        if valid_mask.sum() > 0:
            y_val_valid = y_val[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            mape = mean_absolute_percentage_error(y_val_valid, y_pred_valid) * 100
        else:
            mape = 999.99  # Fallback if no valid samples
        
        mae = mean_absolute_error(y_val, y_pred)
        
        logger.info(f"Global model training complete!")
        logger.info(f"  Validation MAPE: {mape:.2f}%")
        logger.info(f"  Validation MAE: ${mae:.2f}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X, _ = self.prepare_features(df)
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
    
    def save(self, path: str):
        """Save model"""
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Global model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'GlobalModel':
        """Load model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.feature_cols = model_data['feature_cols']
        
        logger.info(f"Global model loaded from {path}")
        return instance


class LocalAdapter:
    """Store-specific residual models for local adaptation"""
    
    def __init__(self, global_model: GlobalModel, config: ModelConfig):
        self.global_model = global_model
        self.config = config
        self.store_adapters = {}  # Store-specific adjustments
    
    def fit_store_adapters(self, train_df: pd.DataFrame, 
                          min_data_days: int = 180) -> 'LocalAdapter':
        """Fit local adapters for data-rich stores"""
        logger.info("Fitting local adapters for data-rich stores...")
        
        # Get global predictions first
        global_preds = self.global_model.predict(train_df)
        train_df['global_pred'] = global_preds
        train_df['residual'] = train_df['sales_amount'] - global_preds
        
        # For each store with enough data, calculate adjustment factors
        for store_id, group in train_df.groupby('store_id'):
            # Check if store has enough history
            if len(group) >= min_data_days:
                # Simple adjustment: mean residual and std
                adapter = {
                    'mean_residual': group['residual'].mean(),
                    'std_residual': group['residual'].std(),
                    'data_points': len(group),
                    'mape_improvement': None  # Calculate later
                }
                self.store_adapters[store_id] = adapter
        
        logger.info(f"Created local adapters for {len(self.store_adapters)} stores")
        return self
    
    def predict_with_adaptation(self, df: pd.DataFrame, 
                                shrinkage_factor: float = 0.7) -> np.ndarray:
        """Predict with Bayesian shrinkage to global baseline"""
        global_preds = self.global_model.predict(df)
        
        predictions = []
        for i, (idx, row) in enumerate(df.iterrows()):
            store_id = row['store_id']
            global_pred = global_preds[i]  # Use enumeration index, not dataframe index
            
            if store_id in self.store_adapters:
                # Apply local adjustment with shrinkage
                adapter = self.store_adapters[store_id]
                local_adjustment = adapter['mean_residual']
                
                # Bayesian shrinkage: more shrinkage for stores with less data
                data_confidence = min(adapter['data_points'] / self.config.data_rich_threshold_days, 1.0)
                shrinkage = shrinkage_factor * data_confidence
                
                adjusted_pred = global_pred + (shrinkage * local_adjustment)
            else:
                # No adapter: use pure global prediction
                adjusted_pred = global_pred
            
            predictions.append(max(0, adjusted_pred))  # Ensure non-negative
        
        return np.array(predictions)
    
    def save(self, path: str):
        """Save local adapters"""
        with open(path, 'wb') as f:
            pickle.dump(self.store_adapters, f)
        logger.info(f"Local adapters saved to {path}")
    
    @classmethod
    def load(cls, path: str, global_model: GlobalModel, config: ModelConfig) -> 'LocalAdapter':
        """Load local adapters"""
        with open(path, 'rb') as f:
            store_adapters = pickle.load(f)
        
        instance = cls(global_model, config)
        instance.store_adapters = store_adapters
        
        logger.info(f"Local adapters loaded from {path}")
        return instance


class ModelTrainingPipeline:
    """Complete model training pipeline"""
    
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        self.model_config = model_config
        self.data_config = data_config
        self.splitter = DataSplitter()
        self.global_model = None
        self.local_adapter = None
        self.clusterer = None
    
    def load_features(self, version: str = 'v1') -> pd.DataFrame:
        """Load features from feature store"""
        feature_path = Path(self.data_config.feature_store_dir) / f'features_{version}.pkl'
        df = pd.read_pickle(feature_path)
        logger.info(f"Loaded {len(df):,} records from feature store")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test"""
        return self.splitter.time_based_split(
            df,
            train_ratio=self.data_config.train_ratio,
            val_ratio=self.data_config.val_ratio,
            test_ratio=self.data_config.test_ratio
        )
    
    def train_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train global model and local adapters"""
        # Train global model
        logger.info("="*60)
        logger.info("Phase 1: Training Global Model")
        logger.info("="*60)
        
        self.global_model = GlobalModel(self.model_config)
        self.global_model.train(train_df, val_df)
        
        # Train local adapters
        logger.info("\n" + "="*60)
        logger.info("Phase 2: Fitting Local Adapters")
        logger.info("="*60)
        
        self.local_adapter = LocalAdapter(self.global_model, self.model_config)
        self.local_adapter.fit_store_adapters(
            train_df, 
            min_data_days=self.model_config.min_data_days
        )
    
    def evaluate_models(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate models on test set"""
        logger.info("\n" + "="*60)
        logger.info("Phase 3: Model Evaluation")
        logger.info("="*60)
        
        # Global model only
        global_preds = self.global_model.predict(test_df)
        
        # Filter out any zero or negative actual values for MAPE calculation
        y_true = test_df['sales_amount'].values
        
        # Create mask for valid values (non-zero actuals)
        valid_mask = y_true > 0
        
        if valid_mask.sum() == 0:
            logger.error("No valid sales values found in test set!")
            return {'error': 'No valid data'}
        
        # Calculate metrics only on valid values
        y_true_valid = y_true[valid_mask]
        global_preds_valid = global_preds[valid_mask]
        
        global_mape = mean_absolute_percentage_error(y_true_valid, global_preds_valid) * 100
        
        # With local adaptation
        adapted_preds = self.local_adapter.predict_with_adaptation(test_df)
        adapted_preds_valid = adapted_preds[valid_mask]
        adapted_mape = mean_absolute_percentage_error(y_true_valid, adapted_preds_valid) * 100
        
        results = {
            'global_model': {
                'mape': global_mape,
                'mae': mean_absolute_error(y_true_valid, global_preds_valid),
                'rmse': np.sqrt(mean_squared_error(y_true_valid, global_preds_valid))
            },
            'adapted_model': {
                'mape': adapted_mape,
                'mae': mean_absolute_error(y_true_valid, adapted_preds_valid),
                'rmse': np.sqrt(mean_squared_error(y_true_valid, adapted_preds_valid))
            },
            'improvement': {
                'mape_reduction': global_mape - adapted_mape
            },
            'evaluation_stats': {
                'total_samples': len(test_df),
                'valid_samples': valid_mask.sum(),
                'zero_sales_samples': (~valid_mask).sum()
            }
        }
        
        logger.info(f"\nGlobal Model Performance:")
        logger.info(f"  MAPE: {global_mape:.2f}%")
        logger.info(f"  MAE: ${results['global_model']['mae']:.2f}")
        
        logger.info(f"\nAdapted Model Performance:")
        logger.info(f"  MAPE: {adapted_mape:.2f}%")
        logger.info(f"  MAE: ${results['adapted_model']['mae']:.2f}")
        logger.info(f"  Improvement: {results['improvement']['mape_reduction']:.2f}% MAPE reduction")
        
        logger.info(f"\nEvaluation Stats:")
        logger.info(f"  Valid samples: {valid_mask.sum()} / {len(test_df)}")
        
        return results
    
    def save_models(self, version: str = 'v1'):
        """Save all models"""
        model_dir = Path(self.data_config.model_dir)
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save global model
        global_path = model_dir / f'global_model_{version}.pkl'
        self.global_model.save(str(global_path))
        
        # Save local adapters
        adapter_path = model_dir / f'local_adapters_{version}.pkl'
        self.local_adapter.save(str(adapter_path))
        
        logger.info(f"Models saved to {model_dir}")
    
    def run(self, version: str = 'v1') -> Dict:
        """Run complete training pipeline"""
        logger.info("="*60)
        logger.info("Starting Model Training Pipeline")
        logger.info("="*60)
        
        # Load data
        df = self.load_features(version)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Train models
        self.train_models(train_df, val_df)
        
        # Evaluate
        results = self.evaluate_models(test_df)
        
        # Save models
        self.save_models(version)
        
        # Save results
        results_path = Path(self.data_config.model_dir) / f'training_results_{version}.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results_clean = convert_to_python_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        logger.info("="*60)
        logger.info("Model Training Pipeline Complete!")
        logger.info("="*60)
        
        return results


if __name__ == '__main__':
    try:
        from config import PlatformConfig
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "02_config.py")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        PlatformConfig = config_module.PlatformConfig
    
    config = PlatformConfig.create_default()
    pipeline = ModelTrainingPipeline(config.model, config.data)
    
    results = pipeline.run(version='v1')
    print(f"\nTraining complete! Results saved to models/training_results_v1.json")
