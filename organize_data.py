#!/usr/bin/env python3
"""
Script to organize training data into separate train/validation/test directories.
This prevents data contamination and ensures proper evaluation.
"""

import os
import shutil
import pickle
import random
import numpy as np
from pathlib import Path

def organize_training_data(
    source_dir: str = "training_grids",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Organize training data into separate directories.
    
    Args:
        source_dir: Directory containing all grid files
        train_ratio: Proportion for training
        val_ratio: Proportion for validation  
        test_ratio: Proportion for testing
        seed: Random seed for reproducible splits
    """
    random.seed(seed)
    
    # Create directories
    train_dir = Path("data/train")
    val_dir = Path("data/validation") 
    test_dir = Path("data/test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load all grid files
    source_path = Path(source_dir)
    grid_files = list(source_path.glob("*.pkl"))
    
    if not grid_files:
        print(f"No grid files found in {source_dir}")
        return
    
    print(f"Found {len(grid_files)} grid files")
    
    # Validate grids (must have 170 non-zero digits)
    valid_files = []
    for grid_file in grid_files:
        try:
            with open(grid_file, 'rb') as f:
                grid = pickle.load(f)
            if np.count_nonzero(grid) == 170:
                valid_files.append(grid_file)
            else:
                print(f"Skipping {grid_file.name}: {np.count_nonzero(grid)} non-zero digits")
        except Exception as e:
            print(f"Error loading {grid_file.name}: {e}")
    
    print(f"Valid grids: {len(valid_files)}")
    
    if not valid_files:
        print("No valid grids found!")
        return
    
    # Shuffle and split
    random.shuffle(valid_files)
    
    n_total = len(valid_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train + n_val]
    test_files = valid_files[n_train + n_val:]
    
    print(f"\nSplitting {n_total} valid grids:")
    print(f"  Training:   {len(train_files)} grids")
    print(f"  Validation: {len(val_files)} grids") 
    print(f"  Test:       {len(test_files)} grids")
    
    # Copy files to appropriate directories
    def copy_files(files, target_dir):
        for file_path in files:
            shutil.copy2(file_path, target_dir / file_path.name)
    
    print("\nCopying files...")
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    print("✅ Data organization complete!")
    print(f"\nDirectory structure:")
    print(f"  data/train/       - {len(train_files)} grids for training")
    print(f"  data/validation/  - {len(val_files)} grids for validation during training")
    print(f"  data/test/        - {len(test_files)} grids for final evaluation")
    
    # Create a summary file
    summary = {
        'total_grids': n_total,
        'train_count': len(train_files),
        'val_count': len(val_files), 
        'test_count': len(test_files),
        'train_files': [f.name for f in train_files],
        'val_files': [f.name for f in val_files],
        'test_files': [f.name for f in test_files],
        'seed': seed
    }
    
    with open('data/split_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nSplit summary saved to: data/split_summary.pkl")

def verify_data_isolation():
    """Verify that train/validation/test sets are completely separate."""
    print("Verifying data isolation...")
    
    train_dir = Path("data/train")
    val_dir = Path("data/validation")
    test_dir = Path("data/test")
    
    if not all(d.exists() for d in [train_dir, val_dir, test_dir]):
        print("❌ Data directories not found. Run organize_training_data() first.")
        return False
    
    # Get file names from each directory
    train_files = set(f.name for f in train_dir.glob("*.pkl"))
    val_files = set(f.name for f in val_dir.glob("*.pkl"))
    test_files = set(f.name for f in test_dir.glob("*.pkl"))
    
    # Check for overlaps
    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files
    
    if train_val_overlap:
        print(f"❌ Overlap between train and validation: {train_val_overlap}")
        return False
    
    if train_test_overlap:
        print(f"❌ Overlap between train and test: {train_test_overlap}")
        return False
    
    if val_test_overlap:
        print(f"❌ Overlap between validation and test: {val_test_overlap}")
        return False
    
    print("✅ Data isolation verified - no overlaps found!")
    print(f"  Train: {len(train_files)} files")
    print(f"  Validation: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize training data into separate directories')
    parser.add_argument('--source', default='training_grids', help='Source directory with grid files')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test data ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verify', action='store_true', help='Verify data isolation')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_data_isolation()
    else:
        organize_training_data(
            source_dir=args.source,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        ) 