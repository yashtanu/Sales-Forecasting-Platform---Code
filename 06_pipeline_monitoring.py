"""
Pipeline 4: Drift Detection & Monitoring
Multi-layered drift detection with automated retraining decisions
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
try:
    from config import DriftConfig, DataConfig
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "02_config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    DriftConfig = config_module.DriftConfig
    DataConfig = config_module.DataConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MAPEMonitor:
    """Monitor MAPE (Mean Absolute Percentage Error) for drift detection"""
    
    def __init__(self, config: DriftConfig):
        self.config = config
    
    def calculate_rolling_mape(self, actuals: pd.Series, predictions: pd.Series,
                               window: int = 7) -> pd.Series:
        """Calculate rolling MAPE"""
        # Calculate absolute percentage error
        ape = np.abs((actuals - predictions) / actuals) * 100
        
        # Rolling mean
        rolling_mape = ape.rolling(window=window, min_periods=1).mean()
        
        return rolling_mape
    
    def detect_mape_drift(self, current_mape: float, baseline_mape: float) -> Dict:
        """Detect drift based on MAPE thresholds"""
        drift_info = {
            'current_mape': current_mape,
            'baseline_mape': baseline_mape,
            'increase_pct': ((current_mape - baseline_mape) / baseline_mape) * 100 if baseline_mape > 0 else 0,
            'drift_detected': False,
            'severity': 'none',
            'action': 'none'
        }
        
        # Check absolute thresholds
        if current_mape >= self.config.mape_emergency_threshold:
            drift_info['drift_detected'] = True
            drift_info['severity'] = 'severe'
            drift_info['action'] = 'retrain_global_emergency'
        elif current_mape >= self.config.mape_alert_threshold:
            drift_info['drift_detected'] = True
            drift_info['severity'] = 'moderate'
            drift_info['action'] = 'retrain_cluster'
        elif current_mape >= self.config.mape_minor_threshold:
            drift_info['drift_detected'] = True
            drift_info['severity'] = 'minor'
            drift_info['action'] = 'alert_only'
        
        # Check relative increase
        if drift_info['increase_pct'] > self.config.mape_increase_threshold * 100:
            drift_info['drift_detected'] = True
            if drift_info['severity'] == 'none':
                drift_info['severity'] = 'minor'
                drift_info['action'] = 'alert_only'
        
        return drift_info


class PSICalculator:
    """Population Stability Index - detects feature distribution shifts"""
    
    @staticmethod
    def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """
        Calculate PSI (Population Stability Index)
        PSI < 0.1: No significant change
        0.1 < PSI < 0.25: Small change
        PSI > 0.25: Significant change - action required
        """
        # Create bins based on expected distribution
        breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        if len(breakpoints) < 2:
            return 0.0  # Not enough variation to calculate PSI
        
        # Calculate expected and actual percentages
        expected_percents = pd.cut(expected, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts(normalize=True)
        actual_percents = pd.cut(actual, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts(normalize=True)
        
        # Align indices
        expected_percents, actual_percents = expected_percents.align(actual_percents, fill_value=0.001)
        
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        expected_percents = expected_percents + epsilon
        actual_percents = actual_percents + epsilon
        
        # Calculate PSI (suppress divide by zero warnings)
        with np.errstate(divide='ignore', invalid='ignore'):
            psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        # Handle NaN or inf results
        if np.isnan(psi_value) or np.isinf(psi_value):
            return 0.0
        
        return float(psi_value)
    
    def detect_feature_drift(self, baseline_features: pd.DataFrame,
                            current_features: pd.DataFrame,
                            threshold: float = 0.25) -> Dict:
        """Detect drift in multiple features"""
        drift_results = {}
        
        numeric_cols = baseline_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in current_features.columns:
                psi = self.calculate_psi(baseline_features[col], current_features[col])
                drift_results[col] = {
                    'psi': psi,
                    'drift_detected': psi > threshold,
                    'severity': 'high' if psi > threshold else ('medium' if psi > 0.1 else 'low')
                }
        
        return drift_results


class CUSUMMonitor:
    """CUSUM (Cumulative Sum) - detects gradual drift"""
    
    def __init__(self, threshold: float = 5.0, drift_threshold: float = 3.0):
        self.threshold = threshold
        self.drift_threshold = drift_threshold
        self.cusum_pos = 0
        self.cusum_neg = 0
    
    def update(self, error: float, target: float = 0) -> Dict:
        """Update CUSUM with new error"""
        deviation = error - target
        
        # Update positive CUSUM
        self.cusum_pos = max(0, self.cusum_pos + deviation - self.drift_threshold)
        
        # Update negative CUSUM
        self.cusum_neg = min(0, self.cusum_neg + deviation + self.drift_threshold)
        
        # Check for drift
        drift_detected = (abs(self.cusum_pos) > self.threshold or 
                         abs(self.cusum_neg) > self.threshold)
        
        return {
            'cusum_pos': self.cusum_pos,
            'cusum_neg': self.cusum_neg,
            'drift_detected': drift_detected,
            'direction': 'positive' if self.cusum_pos > abs(self.cusum_neg) else 'negative'
        }
    
    def reset(self):
        """Reset CUSUM counters"""
        self.cusum_pos = 0
        self.cusum_neg = 0


class DriftDetector:
    """Multi-layered drift detection system"""
    
    def __init__(self, config: DriftConfig):
        self.config = config
        self.mape_monitor = MAPEMonitor(config)
        self.psi_calculator = PSICalculator()
        self.cusum_monitor = CUSUMMonitor(
            threshold=config.cusum_threshold,
            drift_threshold=config.cusum_drift_threshold
        )
    
    def detect_all_drift(self, 
                        baseline_data: pd.DataFrame,
                        current_data: pd.DataFrame,
                        baseline_predictions: pd.Series,
                        current_predictions: pd.Series) -> Dict:
        """Run all drift detection methods"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'methods': {}
        }
        
        # 1. MAPE-based detection - filter out low sales values
        baseline_valid_mask = baseline_data['sales_amount'] > 1.0
        current_valid_mask = current_data['sales_amount'] > 1.0
        
        if baseline_valid_mask.sum() > 0 and current_valid_mask.sum() > 0:
            baseline_mape = mean_absolute_percentage_error(
                baseline_data.loc[baseline_valid_mask, 'sales_amount'],
                baseline_predictions[baseline_valid_mask]
            ) * 100
            current_mape = mean_absolute_percentage_error(
                current_data.loc[current_valid_mask, 'sales_amount'],
                current_predictions[current_valid_mask]
            ) * 100
        else:
            baseline_mape = 999.0
            current_mape = 999.0
        
        mape_drift = self.mape_monitor.detect_mape_drift(current_mape, baseline_mape)
        results['methods']['mape'] = mape_drift
        
        if mape_drift['drift_detected']:
            results['drift_detected'] = True
        
        # 2. PSI-based detection on features
        feature_drift = self.psi_calculator.detect_feature_drift(
            baseline_data, current_data, self.config.psi_threshold
        )
        
        # Count features with significant drift
        drifted_features = sum(1 for v in feature_drift.values() if v['drift_detected'])
        results['methods']['psi'] = {
            'drifted_features': drifted_features,
            'total_features': len(feature_drift),
            'drift_detected': drifted_features > 0,
            'details': feature_drift
        }
        
        if drifted_features > 0:
            results['drift_detected'] = True
        
        # 3. CUSUM-based detection on residuals
        residuals = current_data['sales_amount'] - current_predictions
        mean_residual = residuals.mean()
        
        cusum_result = self.cusum_monitor.update(mean_residual)
        results['methods']['cusum'] = cusum_result
        
        if cusum_result['drift_detected']:
            results['drift_detected'] = True
        
        # Determine overall action
        results['recommended_action'] = self._determine_action(results)
        
        return results
    
    def _determine_action(self, drift_results: Dict) -> str:
        """Determine recommended action based on all drift signals"""
        # Severe drift in MAPE = emergency retrain
        if (drift_results['methods'].get('mape', {}).get('severity') == 'severe'):
            return 'retrain_global_emergency'
        
        # Moderate drift in MAPE or multiple PSI drifts = cluster retrain
        if (drift_results['methods'].get('mape', {}).get('severity') == 'moderate' or
            drift_results['methods'].get('psi', {}).get('drifted_features', 0) >= 5):
            return 'retrain_cluster'
        
        # CUSUM drift = investigate
        if drift_results['methods'].get('cusum', {}).get('drift_detected'):
            return 'investigate_gradual_drift'
        
        # Minor drift = alert only
        if drift_results['drift_detected']:
            return 'alert_only'
        
        return 'none'


class MonitoringPipeline:
    """Complete monitoring and drift detection pipeline"""
    
    def __init__(self, config: DriftConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        self.detector = DriftDetector(config)
    
    def load_predictions_and_actuals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load predictions and actual values for comparison"""
        logger.info("Loading predictions and actuals...")
        
        # Load feature data (contains actuals)
        feature_path = Path(self.data_config.feature_store_dir) / 'features_v1.pkl'
        features_df = pd.read_pickle(feature_path)
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        # For demo, we'll use test set
        # Split into baseline (older) and current (recent)
        features_df = features_df.sort_values('date')
        
        # Baseline: 80% of data
        split_idx = int(len(features_df) * 0.8)
        baseline_df = features_df.iloc[:split_idx].copy()
        current_df = features_df.iloc[split_idx:].copy()
        
        logger.info(f"  • Baseline period: {baseline_df['date'].min()} to {baseline_df['date'].max()}")
        logger.info(f"  • Current period: {current_df['date'].min()} to {current_df['date'].max()}")
        
        return baseline_df, current_df
    
    def generate_predictions(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions using trained model"""
        # Load model
        model_path = Path(self.data_config.model_dir) / 'global_model_v1.pkl'
        
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Prepare features
        try:
            from pipeline_2_training import GlobalModel
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("pipeline_2_training", "04_pipeline_training.py")
            training_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(training_module)
            GlobalModel = training_module.GlobalModel
        
        with open('config.json', 'r') as f:
            config_dict = json.load(f)
        
        try:
            from config import ModelConfig
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", "02_config.py")
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            ModelConfig = config_module.ModelConfig
        
        model_config = ModelConfig(**config_dict['model'])
        
        global_model = GlobalModel(model_config)
        global_model.model = model_data['model']
        global_model.feature_cols = model_data['feature_cols']
        
        predictions = global_model.predict(df)
        
        return pd.Series(predictions, index=df.index)
    
    def analyze_store_health(self, current_df: pd.DataFrame, 
                            current_preds: pd.Series) -> pd.DataFrame:
        """Analyze MAPE by store"""
        logger.info("Analyzing store health...")
        
        current_df = current_df.copy()
        current_df['prediction'] = current_preds
        
        store_metrics = []
        
        for store_id, group in current_df.groupby('store_id'):
            if len(group) < 7:  # Need at least a week of data
                continue
            
            # Filter out very low sales (< $1) before MAPE calculation
            valid_mask = group['sales_amount'] > 1.0
            
            if valid_mask.sum() < 3:  # Need at least 3 valid data points
                continue
            
            group_valid = group[valid_mask]
            
            # Calculate MAPE only on valid values
            mape = mean_absolute_percentage_error(
                group_valid['sales_amount'], 
                group_valid['prediction']
            ) * 100
            
            mae = np.mean(np.abs(group_valid['sales_amount'] - group_valid['prediction']))
            
            store_metrics.append({
                'store_id': store_id,
                'num_days': len(group_valid),
                'mape': mape,
                'mae': mae,
                'avg_actual': group_valid['sales_amount'].mean(),
                'avg_predicted': group_valid['prediction'].mean(),
                'health_status': self._classify_health(mape)
            })
        
        health_df = pd.DataFrame(store_metrics).sort_values('mape', ascending=False)
        
        logger.info(f"  • Total stores analyzed: {len(health_df)}")
        logger.info(f"  • Healthy stores (MAPE<5%): {sum(health_df['health_status']=='healthy')}")
        logger.info(f"  • Warning stores (5%≤MAPE<8%): {sum(health_df['health_status']=='warning')}")
        logger.info(f"  • Critical stores (MAPE≥8%): {sum(health_df['health_status']=='critical')}")
        
        return health_df
    
    def _classify_health(self, mape: float) -> str:
        """Classify store health based on MAPE"""
        if mape < self.config.mape_minor_threshold:
            return 'healthy'
        elif mape < self.config.mape_alert_threshold:
            return 'warning'
        else:
            return 'critical'
    
    def run(self):
        """Run complete monitoring pipeline"""
        logger.info("="*70)
        logger.info(" STARTING DRIFT DETECTION & MONITORING PIPELINE")
        logger.info("="*70)
        
        # Load data
        baseline_df, current_df = self.load_predictions_and_actuals()
        
        # Generate predictions
        logger.info("\nGenerating predictions...")
        baseline_preds = self.generate_predictions(baseline_df)
        current_preds = self.generate_predictions(current_df)
        
        # Run drift detection
        logger.info("\nRunning drift detection...")
        drift_results = self.detector.detect_all_drift(
            baseline_df, current_df,
            baseline_preds, current_preds
        )
        
        # Analyze store health
        store_health = self.analyze_store_health(current_df, current_preds)
        
        # Save results
        self.save_results(drift_results, store_health)
        
        # Generate report
        self.generate_report(drift_results, store_health)
        
        logger.info("="*70)
        logger.info(" MONITORING PIPELINE COMPLETE!")
        logger.info("="*70)
        
        return drift_results, store_health
    
    def save_results(self, drift_results: Dict, store_health: pd.DataFrame):
        """Save monitoring results"""
        monitoring_dir = Path('monitoring')
        monitoring_dir.mkdir(exist_ok=True, parents=True)
        
        # Save drift detection results
        drift_path = monitoring_dir / 'drift_detection_report.json'
        with open(drift_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            clean_results = self._clean_for_json(drift_results)
            json.dump(clean_results, f, indent=2)
        logger.info(f"\n✓ Saved drift detection report to {drift_path}")
        
        # Save store health dashboard
        health_path = monitoring_dir / 'store_health_dashboard.csv'
        store_health.to_csv(health_path, index=False)
        logger.info(f"✓ Saved store health dashboard to {health_path}")
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_report(self, drift_results: Dict, store_health: pd.DataFrame):
        """Generate monitoring report"""
        monitoring_dir = Path('monitoring')
        
        report = []
        report.append("="*70)
        report.append(" DRIFT DETECTION & MONITORING REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n" + "="*70)
        report.append(" DRIFT DETECTION SUMMARY")
        report.append("="*70)
        
        report.append(f"\nOverall Drift Status: {'DRIFT DETECTED' if drift_results['drift_detected'] else '✅ NO DRIFT'}")
        report.append(f"Recommended Action: {drift_results['recommended_action'].upper()}")
        
        # MAPE Analysis
        mape_info = drift_results['methods']['mape']
        report.append(f"\n1. MAPE-Based Detection:")
        report.append(f"   • Baseline MAPE: {mape_info['baseline_mape']:.2f}%")
        report.append(f"   • Current MAPE: {mape_info['current_mape']:.2f}%")
        report.append(f"   • Increase: {mape_info['increase_pct']:.2f}%")
        report.append(f"   • Severity: {mape_info['severity'].upper()}")
        report.append(f"   • Action: {mape_info['action']}")
        
        # PSI Analysis
        psi_info = drift_results['methods']['psi']
        report.append(f"\n2. PSI-Based Detection (Feature Drift):")
        report.append(f"   • Features Analyzed: {psi_info['total_features']}")
        report.append(f"   • Features with Drift: {psi_info['drifted_features']}")
        report.append(f"   • Drift Detected: {'Yes' if psi_info['drift_detected'] else 'No'}")
        
        # CUSUM Analysis
        cusum_info = drift_results['methods']['cusum']
        report.append(f"\n3. CUSUM-Based Detection (Gradual Drift):")
        report.append(f"   • Positive CUSUM: {cusum_info['cusum_pos']:.2f}")
        report.append(f"   • Negative CUSUM: {cusum_info['cusum_neg']:.2f}")
        report.append(f"   • Drift Direction: {cusum_info['direction']}")
        report.append(f"   • Drift Detected: {'Yes' if cusum_info['drift_detected'] else 'No'}")
        
        # Store Health
        report.append("\n" + "="*70)
        report.append(" STORE HEALTH ANALYSIS")
        report.append("="*70)
        
        health_counts = store_health['health_status'].value_counts()
        report.append(f"\nStores by Health Status:")
        for status in ['healthy', 'warning', 'critical']:
            count = health_counts.get(status, 0)
            pct = (count / len(store_health) * 100) if len(store_health) > 0 else 0
            report.append(f"   • {status.capitalize()}: {count} ({pct:.1f}%)")
        
        # Top critical stores
        critical_stores = store_health[store_health['health_status'] == 'critical'].head(5)
        if len(critical_stores) > 0:
            report.append(f"\nTop 5 Critical Stores (Highest MAPE):")
            for _, row in critical_stores.iterrows():
                report.append(f"   • {row['store_id']}: MAPE={row['mape']:.2f}%, MAE=${row['mae']:.2f}")
        
        # Recommendations
        report.append("\n" + "="*70)
        report.append(" RECOMMENDATIONS")
        report.append("="*70)
        
        recommendations = self._generate_recommendations(drift_results, store_health)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"\n{i}. {rec}")
        
        report.append("\n" + "="*70)
        
        # Save report
        report_text = "\n".join(report)
        report_path = monitoring_dir / 'retraining_recommendations.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✓ Saved recommendations to {report_path}")
        
        # Print to console
        print("\n" + report_text)
    
    def _generate_recommendations(self, drift_results: Dict, 
                                 store_health: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        action = drift_results['recommended_action']
        
        if action == 'retrain_global_emergency':
            recommendations.append(
                "URGENT: Retrain global model immediately. MAPE exceeded 12% threshold."
            )
            recommendations.append(
                "Conduct human review to identify root cause (data quality, market changes, etc.)"
            )
        elif action == 'retrain_cluster':
            critical_stores = store_health[store_health['health_status'] == 'critical']['store_id'].tolist()
            recommendations.append(
                f"Retrain models for affected store cluster ({len(critical_stores)} stores)"
            )
            recommendations.append(
                f"Focus on stores: {', '.join(critical_stores[:5])}" + 
                (" and others" if len(critical_stores) > 5 else "")
            )
        elif action == 'investigate_gradual_drift':
            recommendations.append(
                "Investigate gradual drift detected by CUSUM. Monitor for next 7 days."
            )
        elif action == 'alert_only':
            recommendations.append(
                "Continue monitoring. Minor drift detected but within acceptable thresholds."
            )
        else:
            recommendations.append(
                "No action required. Models performing within expected ranges."
            )
        
        # Additional recommendations based on PSI
        psi_drifted = drift_results['methods']['psi']['drifted_features']
        if psi_drifted >= 3:
            recommendations.append(
                f"Feature drift detected in {psi_drifted} features. Review data pipeline for quality issues."
            )
        
        # Store-specific recommendations
        critical_count = sum(store_health['health_status'] == 'critical')
        if critical_count > 10:
            recommendations.append(
                f"{critical_count} stores in critical state. Consider cluster-level intervention."
            )
        
        return recommendations


if __name__ == '__main__':
    # Load configuration
    with open('config.json', 'r') as f:
        config_dict = json.load(f)
    
    drift_config = DriftConfig(**config_dict['drift'])
    data_config = DataConfig(**config_dict['data'])
    
    # Run monitoring pipeline
    pipeline = MonitoringPipeline(drift_config, data_config)
    drift_results, store_health = pipeline.run()
    
    print("\nMonitoring complete! Check the 'monitoring' folder for results.")
