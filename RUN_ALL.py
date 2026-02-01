"""
Master Runner Script
Executes all pipelines in the correct order
"""

import subprocess
import sys
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_header(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}")
        print(f"Error code: {e.returncode}")
        return False

def main():
    """Run all pipeline scripts in order"""
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "SALES FORECASTING PLATFORM" + " "*27 + "║")
    print("║" + " "*12 + "Complete Pipeline Execution" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nThis script will run all 6 pipeline steps in order:")
    print("  1. Data Generation")
    print("  2. Configuration Setup")
    print("  3. Data Ingestion & Feature Engineering")
    print("  4. Model Training")
    print("  5. Inference & Forecasting")
    print("  6. Drift Detection & Monitoring")
    
    print("\nEstimated total time: 15-25 minutes")
    print("\nPress Ctrl+C at any time to stop.\n")
    
    input("Press Enter to begin...")
    
    total_start = time.time()
    
    # Pipeline steps
    steps = [
        ("01_generate_data.py", "Generate Synthetic Data"),
        ("02_config.py", "Setup Configuration"),
        ("03_pipeline_ingestion.py", "Data Ingestion & Feature Engineering"),
        ("04_pipeline_training.py", "Model Training"),
        ("05_pipeline_inference.py", "Inference & Forecasting"),
        ("06_pipeline_monitoring.py", "Drift Detection & Monitoring")
    ]
    
    results = []
    
    for i, (script, description) in enumerate(steps, 1):
        success = run_script(script, f"{i}/6 - {description}")
        results.append((description, success))
        
        if not success:
            print("\n⚠️  Pipeline stopped due to error.")
            print("Fix the error and run this script again, or run individual scripts.")
            break
        
        if i < len(steps):
            print("\n⏸️  Pausing for 2 seconds before next step...")
            time.sleep(2)
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print_header("PIPELINE EXECUTION SUMMARY")
    
    print("Results:")
    for i, (description, success) in enumerate(results, 1):
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {i}. {description}: {status}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "🎉" * 35)
        print("\n✅ ALL PIPELINES COMPLETED SUCCESSFULLY!")
        print(f"\nTotal execution time: {total_elapsed/60:.1f} minutes")
        print("\nYour sales forecasting platform is ready!")
        print("\nNext steps:")
        print("  • Check 'outputs/' for forecasts")
        print("  • Check 'monitoring/' for drift reports")
        print("  • Check 'models/' for trained models")
        print("  • Review README.md for detailed documentation")
        print("\n" + "🎉" * 35)
    else:
        print("\n⚠️  Some pipelines failed. Please review the errors above.")
        print("You can re-run individual scripts or this master script after fixing issues.")
    
    print("\n" + "="*70)
    print(" End of execution")
    print("="*70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline execution interrupted by user.")
        print("You can resume by running individual scripts or this master script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
