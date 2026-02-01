"""
Pipeline 1: Data Ingestion & Feature Engineering
Handles data loading, validation, cleaning, and feature creation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
try:
    from config import DataConfig, ModelConfig
except ImportError:
    # If running from different directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "02_config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    DataConfig = config_module.DataConfig
    ModelConfig = config_module.ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality"""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Check if required columns exist"""
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        return True
    
    @staticmethod
    def detect_outliers(series: pd.Series, n_std: float = 3.0) -> pd.Series:
        """Detect outliers using z-score method"""
        mean = series.mean()
        std = series.std()
        z_scores = np.abs((series - mean) / std)
        return z_scores > n_std
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict:
        """Comprehensive data quality checks"""
        report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for negative sales
        if 'sales_amount' in df.columns:
            report['negative_sales'] = (df['sales_amount'] < 0).sum()
        
        return report


class FeatureEngineer:
    """Feature engineering for sales forecasting"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic temporal
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Season encoding (circular)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Days to/from major holidays
        df['days_to_black_friday'] = self._days_to_holiday(df['date'], month=11, day=25)
        df['days_to_new_year'] = self._days_to_holiday(df['date'], month=1, day=1)
        
        return df
    
    def _days_to_holiday(self, dates: pd.Series, month: int, day: int) -> pd.Series:
        """Calculate days to next occurrence of a holiday"""
        result = []
        for date in dates:
            holiday = datetime(date.year, month, day)
            if date > holiday:
                holiday = datetime(date.year + 1, month, day)
            days = (holiday - date).days
            result.append(days)
        return pd.Series(result, index=dates.index)
    
    def create_lag_features(self, df: pd.DataFrame, 
                           group_cols: List[str] = ['store_id'],
                           target_col: str = 'sales_amount') -> pd.DataFrame:
        """Create lag features for time series"""
        df = df.copy()
        df = df.sort_values(group_cols + ['date'])
        
        for lag in self.config.lag_features:
            df[f'{target_col}_lag_{lag}d'] = df.groupby(group_cols)[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               group_cols: List[str] = ['store_id'],
                               target_col: str = 'sales_amount') -> pd.DataFrame:
        """Create rolling statistics"""
        df = df.copy()
        df = df.sort_values(group_cols + ['date'])
        
        for window in self.config.rolling_windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}d'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Rolling std
            df[f'{target_col}_rolling_std_{window}d'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            
            # Rolling max
            df[f'{target_col}_rolling_max_{window}d'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )
            
            # Rolling min
            df[f'{target_col}_rolling_min_{window}d'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )
        
        return df
    
    def create_hierarchical_features(self, df: pd.DataFrame,
                                    stores_df: pd.DataFrame,
                                    brands_df: pd.DataFrame) -> pd.DataFrame:
        """Create hierarchical embeddings and aggregations"""
        df = df.copy()
        
        # Get columns to merge (avoid duplicates)
        store_cols = ['store_id', 'location_type', 'store_size_sqm', 'tier']
        # country already exists in sales data
        
        brand_cols_to_merge = []
        for col in ['category', 'price_tier', 'avg_price_point']:
            if col not in df.columns:
                brand_cols_to_merge.append(col)
        brand_cols_to_merge.insert(0, 'brand_id')
        
        # Merge store metadata
        df = df.merge(stores_df[store_cols], on='store_id', how='left', suffixes=('', '_store'))
        
        # Merge brand metadata
        if len(brand_cols_to_merge) > 1:
            df = df.merge(brands_df[brand_cols_to_merge], on='brand_id', how='left', suffixes=('', '_brand'))
        
        # Label encode categorical features
        from sklearn.preprocessing import LabelEncoder
        
        cat_cols = []
        for col in ['country', 'location_type', 'tier', 'category', 'price_tier']:
            if col in df.columns:
                cat_cols.append(col)
        
        for col in cat_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Store cluster features (aggregate by similar stores)
        if 'tier' in df.columns and 'location_type' in df.columns:
            df['store_cluster'] = df['tier'].astype(str) + '_' + df['location_type'].astype(str)
        
        return df
    
    def create_all_features(self, sales_df: pd.DataFrame,
                           stores_df: pd.DataFrame,
                           brands_df: pd.DataFrame,
                           external_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create all features for modeling"""
        logger.info("Creating all features...")
        
        # Temporal features
        logger.info("  - Temporal features")
        df = self.create_temporal_features(sales_df)
        
        # Hierarchical features
        logger.info("  - Hierarchical features")
        df = self.create_hierarchical_features(df, stores_df, brands_df)
        
        # Merge external features
        if external_df is not None:
            logger.info("  - External features")
            df['date'] = pd.to_datetime(df['date'])
            external_df['date'] = pd.to_datetime(external_df['date'])
            df = df.merge(external_df, on=['date', 'country'], how='left')
        
        # Sort for time series features
        df = df.sort_values(['store_id', 'date'])
        
        # Lag features (only for stores with enough history)
        logger.info("  - Lag features")
        df = self.create_lag_features(df)
        
        # Rolling features
        logger.info("  - Rolling features")
        df = self.create_rolling_features(df)
        
        # Fill NaN values from lag/rolling features
        # For new stores, use brand/cluster means
        feature_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        for col in feature_cols:
            df[col] = df.groupby('brand_id')[col].transform(
                lambda x: x.fillna(x.mean())
            )
            df[col] = df[col].fillna(0)  # If still NaN, fill with 0
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df


class DataIngestionPipeline:
    """Main data ingestion pipeline"""
    
    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer(model_config)
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files"""
        logger.info("Loading raw data...")
        
        data = {}
        files = {
            'stores': self.data_config.stores_file,
            'brands': self.data_config.brands_file,
            'sales': self.data_config.sales_file,
            'external': self.data_config.external_file
        }
        
        for name, filename in files.items():
            filepath = Path(self.data_config.data_dir) / filename
            data[name] = pd.read_csv(filepath)
            logger.info(f"  Loaded {name}: {len(data[name]):,} rows")
        
        return data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate all datasets"""
        logger.info("Validating data quality...")
        
        # Sales data validation
        sales_quality = self.validator.check_data_quality(data['sales'])
        logger.info(f"  Sales data quality: {sales_quality}")
        
        return True
    
    def process_and_engineer_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process data and create features"""
        logger.info("Processing data and engineering features...")
        
        # Create features
        featured_df = self.feature_engineer.create_all_features(
            sales_df=data['sales'],
            stores_df=data['stores'],
            brands_df=data['brands'],
            external_df=data['external']
        )
        
        return featured_df
    
    def save_to_feature_store(self, df: pd.DataFrame, version: str = 'v1'):
        """Save features to feature store"""
        output_dir = Path(self.data_config.feature_store_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save as pickle (parquet needs extra dependencies)
        output_path = output_dir / f'features_{version}.pkl'
        df.to_pickle(output_path)
        
        logger.info(f"Features saved to {output_path}")
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'num_records': len(df),
            'num_features': len(df.columns),
            'features': list(df.columns),
            'stores': df['store_id'].nunique(),
            'brands': df['brand_id'].nunique(),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            }
        }
        
        import json
        metadata_path = output_dir / f'metadata_{version}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return output_path
    
    def run(self, version: str = 'v1') -> str:
        """Run the complete data ingestion pipeline"""
        logger.info("="*60)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("="*60)
        
        # Load data
        data = self.load_raw_data()
        
        # Validate
        self.validate_data(data)
        
        # Process and engineer features
        featured_df = self.process_and_engineer_features(data)
        
        # Save to feature store
        output_path = self.save_to_feature_store(featured_df, version)
        
        logger.info("="*60)
        logger.info("Data Ingestion Pipeline Complete!")
        logger.info("="*60)
        
        return str(output_path)


if __name__ == '__main__':
    # Import config
    try:
        from config import PlatformConfig
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "02_config.py")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        PlatformConfig = config_module.PlatformConfig
    
    config = PlatformConfig.create_default()
    pipeline = DataIngestionPipeline(config.data, config.model)
    
    output_path = pipeline.run(version='v1')
    print(f"\nFeatures ready at: {output_path}")
