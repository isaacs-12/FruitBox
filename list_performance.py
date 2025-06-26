#!/usr/bin/env python3
"""
Script to list all performance data files and plots for different models.
"""

import os
import glob
import json
from datetime import datetime

def list_performance_data():
    """List all performance data files and their statistics."""
    print("=" * 80)
    print("PERFORMANCE DATA SUMMARY")
    print("=" * 80)
    
    # Find all performance data files
    data_files = glob.glob("performance_data_*.json")
    
    if not data_files:
        print("No performance data files found.")
        return
    
    print(f"Found {len(data_files)} performance data files:\n")
    
    for data_file in sorted(data_files):
        model_name = data_file.replace("performance_data_", "").replace(".json", "")
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            if not data['attempts']:
                print(f"📊 {model_name}: No attempts recorded")
                continue
            
            total_attempts = len(data['attempts'])
            successful_attempts = sum(data['threshold_met'])
            success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
            avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
            best_score = max(data['scores']) if data['scores'] else 0
            worst_score = min(data['scores']) if data['scores'] else 0
            
            # Check if plot exists
            plot_file = f"performance_plot_{model_name}.png"
            plot_exists = "📈" if os.path.exists(plot_file) else "❌"
            
            print(f"{plot_exists} {model_name}:")
            print(f"   Attempts: {total_attempts}")
            print(f"   Success Rate: {success_rate:.1f}% ({successful_attempts}/{total_attempts})")
            print(f"   Average Score: {avg_score:.1f}")
            print(f"   Best Score: {best_score}")
            print(f"   Worst Score: {worst_score}")
            
            # Show recent attempts
            if data['timestamps']:
                latest_timestamp = data['timestamps'][-1]
                try:
                    latest_time = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                    print(f"   Last Run: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    print(f"   Last Run: {latest_timestamp}")
            
            print()
            
        except Exception as e:
            print(f"❌ {model_name}: Error reading data - {e}")
            print()

def list_available_models():
    """List all available trained models."""
    print("=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)
    
    model_files = glob.glob("models/*.zip")
    
    if not model_files:
        print("No trained models found in models/ directory.")
        return
    
    print(f"Found {len(model_files)} trained models:\n")
    
    for model_file in sorted(model_files):
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        
        # Check if this model has performance data
        data_file = f"performance_data_{model_name}.json"
        has_data = "📊" if os.path.exists(data_file) else "❌"
        
        print(f"{has_data} {model_name} ({file_size:.1f} MB)")
    
    print()

if __name__ == "__main__":
    list_available_models()
    list_performance_data()
    
    print("=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("To run with a specific model:")
    print("  make run-model MODEL=models/model_20250626_102548.zip")
    print("  python main.py --model models/model_20250626_102548.zip")
    print()
    print("To run with verbose output:")
    print("  make run-model-verbose MODEL=models/model_20250626_102548.zip")
    print("  python main.py --model models/model_20250626_102548.zip --verbose")
    print()
    print("To run with test image:")
    print("  python main.py --model models/model_20250626_102548.zip --test") 