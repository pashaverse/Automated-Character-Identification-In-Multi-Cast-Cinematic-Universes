import os
import shutil
from pathlib import Path
from collections import defaultdict

# Configuration
DATASET_ROOT = "dataset"
OUTPUT_ROOT = "finalised_dataset"
CHARACTERS = ["dustin", "eleven", "hopper", "lucas", "steve"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

# Mapping of actor folder names to characters for S2/scraped
ACTOR_TO_CHARACTER = {
    "millie_bobby_brown": "eleven",
    "millie bobby brown": "eleven",
    "gaten_matarazzo": "dustin",
    "gaten matarazzo": "dustin",
    "caleb_mclaughlin": "lucas",
    "caleb mclaughlin": "lucas",
    "joe_keery": "steve",
    "joe keery": "steve",
    "david_harbour": "hopper",
    "david harbour": "hopper",
}

def ensure_output_dirs():
    """Create output directory structure."""
    for character in CHARACTERS:
        char_dir = Path(OUTPUT_ROOT) / character
        char_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {char_dir}")

def is_image_file(filename):
    """Check if file is an image based on extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

def get_character_from_actor_folder(folder_name):
    """Map actor folder name to character name."""
    # Normalize: lowercase, remove "stranger things season 2" suffix, strip whitespace
    normalized = folder_name.lower()
    normalized = normalized.replace("stranger things season 2", "").strip()
    
    # Try exact match
    if normalized in ACTOR_TO_CHARACTER:
        return ACTOR_TO_CHARACTER[normalized]
    
    # Try with underscores instead of spaces
    normalized_underscores = normalized.replace(" ", "_")
    if normalized_underscores in ACTOR_TO_CHARACTER:
        return ACTOR_TO_CHARACTER[normalized_underscores]
    
    return None

def copy_images_from_path(source_path, character, season, counts):
    """Copy images from source to output folder with standardized naming."""
    source_path = Path(source_path)
    if not source_path.exists():
        return
    
    output_dir = Path(OUTPUT_ROOT) / character
    files_copied = 0
    
    # Get starting index for this character
    index = counts[character] + 1
    
    # Iterate through files in source
    for filename in os.listdir(source_path):
        filepath = source_path / filename
        
        # Skip directories and non-image files
        if not filepath.is_file() or not is_image_file(filename):
            continue
        
        # Create standardized filename: {character}_{season}_{index}.{ext}
        ext = Path(filename).suffix.lower()
        new_filename = f"{character}_{season}_{index}{ext}"
        output_path = output_dir / new_filename
        
        try:
            shutil.copy2(filepath, output_path)
            counts[character] += 1
            files_copied += 1
            index += 1
        except Exception as e:
            print(f"    Error copying {filename}: {e}")
    
    return files_copied

def consolidate_dataset():
    """Main consolidation function."""
    counts = defaultdict(int)
    
    print("="*60)
    print("DATASET CONSOLIDATION SCRIPT")
    print("="*60)
    print()
    
    # Create output directories
    print("Creating output directory structure...")
    ensure_output_dirs()
    print()
    
    # S1: dataset/S1/sorted/{character}/
    print("Processing S1 (sorted)...")
    s1_total = 0
    for character in CHARACTERS:
        source = Path(DATASET_ROOT) / "S1" / "sorted" / character
        copied = copy_images_from_path(source, character, "S1", counts)
        if copied:
            print(f"  {character}: {copied} images")
            s1_total += copied
    print()
    
    # S2: dataset/S2/cropped_faces/{character}/
    print("Processing S2 (cropped_faces)...")
    s2_cropped_total = 0
    for character in CHARACTERS:
        source = Path(DATASET_ROOT) / "S2" / "cropped_faces" / character
        copied = copy_images_from_path(source, character, "S2", counts)
        if copied:
            print(f"  {character}: {copied} images")
            s2_cropped_total += copied
    print()
    
    # S2: dataset/S2/scraped/ - actor folders mapped to characters
    print("Processing S2 (scraped - actor folders)...")
    s2_scraped_total = 0
    s2_skipped = []
    scraped_path = Path(DATASET_ROOT) / "S2" / "scraped"
    
    if scraped_path.exists():
        for actor_folder in sorted(os.listdir(scraped_path)):
            actor_path = scraped_path / actor_folder
            if not actor_path.is_dir():
                continue
            
            # Map actor folder to character
            character = get_character_from_actor_folder(actor_folder)
            if character:
                copied = copy_images_from_path(actor_path, character, "S2", counts)
                if copied:
                    print(f"  {actor_folder} -> {character}: {copied} images")
                    s2_scraped_total += copied
            else:
                s2_skipped.append(actor_folder)
    
    if s2_skipped:
        print(f"  Skipped (not in actor mapping): {', '.join(s2_skipped)}")
    print()
    
    # S4: dataset/S4/dataset/{character}/
    print("Processing S4...")
    s4_total = 0
    for character in CHARACTERS:
        source = Path(DATASET_ROOT) / "S4" / "dataset" / character
        copied = copy_images_from_path(source, character, "S4", counts)
        if copied:
            print(f"  {character}: {copied} images")
            s4_total += copied
    print()
    
    # S5: dataset/S5/dataset/{character}/
    print("Processing S5...")
    s5_total = 0
    for character in CHARACTERS:
        source = Path(DATASET_ROOT) / "S5" / "dataset" / character
        copied = copy_images_from_path(source, character, "S5", counts)
        if copied:
            print(f"  {character}: {copied} images")
            s5_total += copied
    print()
    
    # Print summary
    print("="*60)
    print("CONSOLIDATION COMPLETE")
    print("="*60)
    print()
    print("Images per character:")
    print("-" * 40)
    total_images = 0
    for character in CHARACTERS:
        count = counts[character]
        total_images += count
        print(f"  {character.capitalize():12} : {count:5} images")
    print("-" * 40)
    print(f"  {'TOTAL':12} : {total_images:5} images")
    print()
    print(f"Output location: {OUTPUT_ROOT}/")
    print()

if __name__ == "__main__":
    consolidate_dataset()
