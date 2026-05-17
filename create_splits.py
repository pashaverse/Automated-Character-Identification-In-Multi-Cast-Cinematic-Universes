"""
Create stratified 70/15/15 train/val/test split from consolidated dataset.
Source: finalised_dataset/{character}/
Output: finalised_dataset/splits/{train,val,test}/{character}/
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
DATASET_ROOT = Path("finalised_dataset")
SPLITS_ROOT = DATASET_ROOT / "splits"
CHARACTERS = ["dustin", "eleven", "hopper", "lucas", "steve"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
RANDOM_STATE = 42

def is_image_file(filename):
    """Check if file is an image based on extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

def create_split_dirs():
    """Create output directory structure for splits."""
    for split_type in ["train", "val", "test"]:
        for character in CHARACTERS:
            split_dir = SPLITS_ROOT / split_type / character
            split_dir.mkdir(parents=True, exist_ok=True)
    print("Created split directory structure.")
    print()

def get_image_files(character):
    """Get all image files for a character from the consolidated dataset."""
    char_dir = DATASET_ROOT / character
    if not char_dir.exists():
        return []
    
    files = []
    for filename in sorted(os.listdir(char_dir)):
        filepath = char_dir / filename
        if filepath.is_file() and is_image_file(filename):
            files.append(filename)
    
    return files

def create_stratified_splits():
    """Create stratified 70/15/15 train/val/test splits."""
    all_counts = {split_type: defaultdict(int) for split_type in ["train", "val", "test"]}
    
    print("="*60)
    print("CREATING STRATIFIED SPLITS (70/15/15)")
    print("="*60)
    print()
    
    # Create output directories
    print("Creating split directory structure...")
    create_split_dirs()
    
    # Process each character
    print("Processing characters:")
    print("-" * 60)
    
    for character in CHARACTERS:
        files = get_image_files(character)
        n_files = len(files)
        
        if n_files == 0:
            print(f"{character.capitalize():12} : SKIP (no images found)")
            continue
        
        # Create stratified labels (all belong to same character class)
        # We stratify to ensure balanced distribution across splits
        stratify_labels = [character] * n_files
        
        # First split: 70% train, 30% temp (for val+test)
        train_files, temp_files = train_test_split(
            files,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=stratify_labels
        )
        
        # Second split: split temp into val and test (15% each of original)
        # 50/50 split of the 30% gives us 15/15
        val_files, test_files = train_test_split(
            temp_files,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=[character] * len(temp_files)
        )
        
        # Copy files to train split
        train_dir = SPLITS_ROOT / "train" / character
        for filename in train_files:
            src = DATASET_ROOT / character / filename
            dst = train_dir / filename
            shutil.copy2(src, dst)
            all_counts["train"][character] += 1
        
        # Copy files to val split
        val_dir = SPLITS_ROOT / "val" / character
        for filename in val_files:
            src = DATASET_ROOT / character / filename
            dst = val_dir / filename
            shutil.copy2(src, dst)
            all_counts["val"][character] += 1
        
        # Copy files to test split
        test_dir = SPLITS_ROOT / "test" / character
        for filename in test_files:
            src = DATASET_ROOT / character / filename
            dst = test_dir / filename
            shutil.copy2(src, dst)
            all_counts["test"][character] += 1
        
        print(f"{character.capitalize():12} : {n_files:5} total → "
              f"train:{len(train_files):4} val:{len(val_files):4} test:{len(test_files):4}")
    
    print()
    print("="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    print()
    
    # Print detailed counts
    print("Per-Character Distribution:")
    print("-" * 60)
    print(f"{'Character':<12} | {'Train':>6} | {'Val':>6} | {'Test':>6} | {'Total':>6}")
    print("-" * 60)
    
    grand_totals = {split_type: 0 for split_type in ["train", "val", "test"]}
    
    for character in CHARACTERS:
        train_count = all_counts["train"][character]
        val_count = all_counts["val"][character]
        test_count = all_counts["test"][character]
        total = train_count + val_count + test_count
        
        grand_totals["train"] += train_count
        grand_totals["val"] += val_count
        grand_totals["test"] += test_count
        
        print(f"{character.capitalize():<12} | {train_count:>6} | {val_count:>6} | {test_count:>6} | {total:>6}")
    
    print("-" * 60)
    total_all = grand_totals["train"] + grand_totals["val"] + grand_totals["test"]
    print(f"{'TOTAL':<12} | {grand_totals['train']:>6} | {grand_totals['val']:>6} | {grand_totals['test']:>6} | {total_all:>6}")
    print("-" * 60)
    print()
    
    # Print percentages
    print("Overall Split Percentages:")
    print("-" * 40)
    if total_all > 0:
        train_pct = (grand_totals["train"] / total_all) * 100
        val_pct = (grand_totals["val"] / total_all) * 100
        test_pct = (grand_totals["test"] / total_all) * 100
        print(f"  Train: {train_pct:6.2f}% ({grand_totals['train']:5} images)")
        print(f"  Val:   {val_pct:6.2f}% ({grand_totals['val']:5} images)")
        print(f"  Test:  {test_pct:6.2f}% ({grand_totals['test']:5} images)")
    print()
    
    print(f"Output location: {SPLITS_ROOT}/")
    print()

if __name__ == "__main__":
    create_stratified_splits()
