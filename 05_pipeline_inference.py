"""
Pipeline 3: Inference & Prediction
Generate forecasts with global model + local adaptation
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_percentage_error
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from other modules
try:
    from config import InferenceConfig, DataConfig, ModelConfig
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "02_config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    InferenceConfig = config_module.InferenceConfig
    DataConfig = config_module.DataConfig
    ModelConfig = config_module.ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ForecastGenerator:
    """Generate future forecasts using trained models"""
    
    def __init__(self, global_model, local_adapter, config: InferenceConfig):
        self.global_model = global_model
        self.local_adapter = local_adapter
        self.config = config
    
    def create_future_dates(self, last_date: str, horizon_days: int = 30) -> pd.DataFrame:
        """Create future date range for forecasting"""
        last_date = pd.to_datetime(last_date)
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        return future_dates
    
    def prepare_forecast_features(self, 
                                  stores_df: pd.DataFrame,
                                  brands_df: pd.DataFrame,
                                  historical_df: pd.DataFrame,
                                  future_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Prepare features for future dates"""
        logger.info("Preparing forecast features...")
        
        forecast_records = []
        
        for store_id in stores_df['store_id'].unique():
            store_info = stores_df[stores_df['store_id'] == store_id].iloc[0]
            
            # Get historical data for this store
            store_hist = historical_df[historical_df['store_id'] == store_id].copy()
            if len(store_hist) == 0:
                continue
            
            store_hist = store_hist.sort_values('date')
            
            for date in future_dates:
                # Create base record - MATCH TRAINING EXACTLY
                record = {
                    'date': date,
                    'store_id': store_id,
                    'brand_id': store_info['brand_id'],
                    'country': store_info['country'],
                    'sales_amount': 0,  # Dummy for model
                }
                
                # Add temporal features (match training exactly)
                record['day_of_week'] = date.dayofweek
                record['day_of_month'] = date.day
                record['month'] = date.month
                record['quarter'] = date.quarter
                record['year'] = date.year
                # Note: week_of_year NOT included - wasn't in training features
                
                # Circular encoding
                record['month_sin'] = np.sin(2 * np.pi * date.month / 12)
                record['month_cos'] = np.cos(2 * np.pi * date.month / 12)
                record['day_of_week_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
                record['day_of_week_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
                
                # Days to holidays
                record['days_to_black_friday'] = self._days_to_holiday(date, 11, 25)
                record['days_to_new_year'] = self._days_to_holiday(date, 1, 1)
                
                # Lag features (from historical data)
                for lag in [7, 14, 30, 365]:
                    lag_date = date - timedelta(days=lag)
                    lag_value = store_hist[store_hist['date'] == lag_date]['sales_amount'].values
                    record[f'sales_amount_lag_{lag}d'] = lag_value[0] if len(lag_value) > 0 else 0
                
                # Rolling features (from historical data)
                for window in [7, 14, 30]:
                    window_start = date - timedelta(days=window)
                    window_data = store_hist[
                        (store_hist['date'] >= window_start) & (store_hist['date'] < date)
                    ]['sales_amount']
                    
                    record[f'sales_amount_rolling_mean_{window}d'] = window_data.mean() if len(window_data) > 0 else 0
                    record[f'sales_amount_rolling_std_{window}d'] = window_data.std() if len(window_data) > 0 else 0
                    record[f'sales_amount_rolling_max_{window}d'] = window_data.max() if len(window_data) > 0 else 0
                    record[f'sales_amount_rolling_min_{window}d'] = window_data.min() if len(window_data) > 0 else 0
                
                # Store metadata features
                record['store_size_sqm'] = store_info['store_size_sqm']
                
                # Brand metadata
                brand_info = brands_df[brands_df['brand_id'] == store_info['brand_id']]
                if len(brand_info) > 0:
                    brand_info = brand_info.iloc[0]
                    record['avg_price_point'] = brand_info['avg_price_point']
                else:
                    record['avg_price_point'] = 500
                
                # External features (simplified - would come from external API)
                record['gdp_growth'] = 3.0
                record['tourism_index'] = 100.0
                record['currency_rate_to_usd'] = 3.67
                record['temperature_celsius'] = 25.0
                record['is_public_holiday'] = 0
                
                # Promotion features
                record['promotion_flag'] = 0
                record['promotion_discount'] = 0.0
                
                # Transaction features (use historical average)
                if len(store_hist) > 0:
                    record['num_transactions'] = int(store_hist['num_transactions'].mean())
                    record['avg_transaction_value'] = store_hist['avg_transaction_value'].mean()
                else:
                    record['num_transactions'] = 0
                    record['avg_transaction_value'] = 0
                
                forecast_records.append(record)
        
        df = pd.DataFrame(forecast_records)
        logger.info(f"Created {len(df)} forecast records")
        
        return df
    
    def _days_to_holiday(self, date, month: int, day: int) -> int:
        """Calculate days to next occurrence of a holiday"""
        holiday = datetime(date.year, month, day)
        if date > holiday:
            holiday = datetime(date.year + 1, month, day)
        return (holiday - date).days
    
    def generate_forecasts(self, 
                          forecast_df: pd.DataFrame,
                          prediction_interval: float = 0.8) -> pd.DataFrame:
        """Generate forecasts with prediction intervals"""
        logger.info("Generating forecasts...")
        
        # Add dummy sales_amount column for the model's prepare_features method
        # (The model expects this column but won't use it during prediction)
        forecast_df['sales_amount'] = 0
        
        # Get predictions from adapted model
        predictions = self.local_adapter.predict_with_adaptation(forecast_df)
        
        # Remove the dummy column
        forecast_df = forecast_df.drop('sales_amount', axis=1)
        
        # Add predictions to dataframe
        forecast_df['predicted_sales'] = predictions
        
        # Calculate prediction intervals (simplified - using residual std)
        # In production, use quantile regression or bootstrap
        prediction_std = np.std(predictions) * 0.3  # Simplified
        z_score = 1.28  # For 80% confidence interval
        
        forecast_df['lower_bound'] = np.maximum(0, predictions - z_score * prediction_std)
        forecast_df['upper_bound'] = predictions + z_score * prediction_std
        
        # Add confidence level
        forecast_df['confidence_level'] = prediction_interval
        
        # Flag prediction type
        forecast_df['prediction_type'] = forecast_df['store_id'].apply(
            lambda x: 'adapted' if x in self.local_adapter.store_adapters else 'global_only'
        )
        
        logger.info("Forecasts generated successfully!")
        
        return forecast_df


class InferencePipeline:
    """Complete inference pipeline"""
    
    def __init__(self, config: InferenceConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        self.global_model = None
        self.local_adapter = None
    
    def load_models(self, version: str = 'v1'):
        """Load trained models"""
        logger.info("Loading trained models...")
        
        model_dir = Path(self.data_config.model_dir)
        
        # Load global model
        global_path = model_dir / f'global_model_{version}.pkl'
        with open(global_path, 'rb') as f:
            model_data = pickle.load(f)
            self.global_model = model_data
        
        # Load local adapters
        adapter_path = model_dir / f'local_adapters_{version}.pkl'
        with open(adapter_path, 'rb') as f:
            adapter_data = pickle.load(f)
        
        # Reconstruct local adapter
        try:
            from pipeline_2_training import LocalAdapter, GlobalModel
        except ImportError:
            # Load from file
            import importlib.util
            spec = importlib.util.spec_from_file_location("pipeline_2_training", "04_pipeline_training.py")
            training_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(training_module)
            LocalAdapter = training_module.LocalAdapter
            GlobalModel = training_module.GlobalModel
        
        config_path = Path('config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_config = ModelConfig(**config_dict['model'])
        
        # Create adapter instance
        try:
            from pipeline_2_training import GlobalModel
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("pipeline_2_training", "04_pipeline_training.py")
            training_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(training_module)
            GlobalModel = training_module.GlobalModel
        
        global_model_obj = GlobalModel(model_config)
        global_model_obj.model = self.global_model['model']
        global_model_obj.feature_cols = self.global_model['feature_cols']
        
        self.local_adapter = LocalAdapter(global_model_obj, model_config)
        self.local_adapter.store_adapters = adapter_data
        
        logger.info("Models loaded successfully!")
        logger.info(f"  • Global model features: {len(self.global_model['feature_cols'])}")
        logger.info(f"  • Local adapters: {len(adapter_data)}")
    
    def load_data(self):
        """Load necessary data"""
        logger.info("Loading data...")
        
        data_dir = Path(self.data_config.data_dir)
        
        stores_df = pd.read_csv(data_dir / self.data_config.stores_file)
        brands_df = pd.read_csv(data_dir / self.data_config.brands_file)
        
        # Load features (historical with actuals)
        feature_path = Path(self.data_config.feature_store_dir) / 'features_v1.pkl'
        historical_df = pd.read_pickle(feature_path)
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        
        logger.info(f"  • Stores: {len(stores_df)}")
        logger.info(f"  • Brands: {len(brands_df)}")
        logger.info(f"  • Historical records: {len(historical_df)}")
        
        return stores_df, brands_df, historical_df
    
    def run(self, horizon_days: int = 30, version: str = 'v1'):
        """Run complete inference pipeline"""
        logger.info("="*70)
        logger.info(" STARTING INFERENCE PIPELINE")
        logger.info("="*70)
        
        # Load models
        self.load_models(version)
        
        # Load data
        stores_df, brands_df, historical_df = self.load_data()
        
        # Create forecast generator
        generator = ForecastGenerator(
            self.global_model,
            self.local_adapter,
            self.config
        )
        
        # Get last date from historical data
        last_date = historical_df['date'].max()
        logger.info(f"\nLast historical date: {last_date}")
        logger.info(f"Forecast horizon: {horizon_days} days")
        
        # Create future dates
        future_dates = generator.create_future_dates(last_date, horizon_days)
        logger.info(f"Forecast period: {future_dates[0]} to {future_dates[-1]}")
        
        # Prepare forecast features
        forecast_df = generator.prepare_forecast_features(
            stores_df, brands_df, historical_df, future_dates
        )
        
        # Generate forecasts
        forecasts = generator.generate_forecasts(
            forecast_df,
            prediction_interval=self.config.confidence_level
        )
        
        # Save forecasts
        self.save_forecasts(forecasts, horizon_days)
        
        # Generate summary
        self.generate_summary(forecasts)
        
        logger.info("="*70)
        logger.info(" INFERENCE PIPELINE COMPLETE!")
        logger.info("="*70)
        
        return forecasts
    
    def save_forecasts(self, forecasts: pd.DataFrame, horizon_days: int):
        """Save forecasts to files"""
        output_dir = Path(self.data_config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Select columns for output
        output_cols = [
            'date', 'store_id', 'brand_id', 'country',
            'predicted_sales', 'lower_bound', 'upper_bound',
            'confidence_level', 'prediction_type'
        ]
        
        # Detailed forecasts
        detailed_path = output_dir / f'forecasts_next_{horizon_days}days.csv'
        forecasts[output_cols].to_csv(detailed_path, index=False)
        logger.info(f"\n✓ Saved detailed forecasts to {detailed_path}")
        
        # Summary by store
        summary = forecasts.groupby(['store_id', 'brand_id', 'country']).agg({
            'predicted_sales': ['sum', 'mean'],
            'prediction_type': 'first'
        }).reset_index()
        summary.columns = ['store_id', 'brand_id', 'country', 
                          'total_forecast', 'avg_daily_forecast', 'prediction_type']
        
        summary_path = output_dir / 'forecast_summary.csv'
        summary.to_csv(summary_path, index=False)
        logger.info(f"Saved forecast summary to {summary_path}")
    
    def generate_summary(self, forecasts: pd.DataFrame):
        """Generate summary report"""
        output_dir = Path(self.data_config.output_dir)
        
        report = []
        report.append("="*70)
        report.append(" FORECAST SUMMARY REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nForecast Period:")
        report.append(f"  • Start: {forecasts['date'].min()}")
        report.append(f"  • End: {forecasts['date'].max()}")
        report.append(f"  • Days: {forecasts['date'].nunique()}")
        
        report.append(f"\nCoverage:")
        report.append(f"  • Total Stores: {forecasts['store_id'].nunique()}")
        report.append(f"  • Total Brands: {forecasts['brand_id'].nunique()}")
        report.append(f"  • Countries: {forecasts['country'].nunique()}")
        
        report.append(f"\nPrediction Types:")
        pred_types = forecasts.groupby('prediction_type')['store_id'].nunique()
        for ptype, count in pred_types.items():
            report.append(f"  • {ptype}: {count} stores")
        
        report.append(f"\nForecast Statistics:")
        report.append(f"  • Total Forecasted Sales: ${forecasts['predicted_sales'].sum():,.2f}")
        report.append(f"  • Average Daily Sales: ${forecasts['predicted_sales'].mean():,.2f}")
        report.append(f"  • Median Daily Sales: ${forecasts['predicted_sales'].median():,.2f}")
        
        # Top stores by forecast
        top_stores = forecasts.groupby('store_id')['predicted_sales'].sum().sort_values(ascending=False).head(5)
        report.append(f"\nTop 5 Stores by Forecast:")
        for store, sales in top_stores.items():
            report.append(f"  • {store}: ${sales:,.2f}")
        
        report.append("\n" + "="*70)
        
        # Save report
        report_text = "\n".join(report)
        report_path = output_dir / 'forecast_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"✓ Saved forecast report to {report_path}")
        
        # Print to console
        print("\n" + report_text)


if __name__ == '__main__':
    # Load configuration
    with open('config.json', 'r') as f:
        config_dict = json.load(f)
    
    inference_config = InferenceConfig(**config_dict['inference'])
    data_config = DataConfig(**config_dict['data'])
    
    # Run inference pipeline
    pipeline = InferencePipeline(inference_config, data_config)
    forecasts = pipeline.run(horizon_days=30, version='v1')
    
    print("\nForecasts generated successfully!")
    print("Check the 'outputs' folder for results.")