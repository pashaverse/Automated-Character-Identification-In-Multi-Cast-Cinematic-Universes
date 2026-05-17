import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


DATASET_ROOT = Path("dataset")
SEASONS = ["S4", "S5"]
CHARACTERS = ["dustin", "eleven", "hopper", "lucas", "steve"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def has_existing_output(output_dir: Path) -> bool:
    return output_dir.exists() and any(child.is_file() for child in output_dir.iterdir())


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return [path for path in sorted(directory.iterdir()) if path.is_file() and is_image_file(path)]


def clip_box(box: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    left = max(0, int(np.floor(box[0])))
    top = max(0, int(np.floor(box[1])))
    right = min(width, int(np.ceil(box[2])))
    bottom = min(height, int(np.ceil(box[3])))

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def process_character(mtcnn: MTCNN, season: str, character: str) -> Dict[str, Any]:
    input_dir = DATASET_ROOT / season / "raw_frames" / character
    output_dir = DATASET_ROOT / season / "dataset" / character

    stats: Dict[str, Any] = {
        "character": character,
        "season": season,
        "total": 0,
        "kept": 0,
        "status": "",
    }

    if has_existing_output(output_dir):
        stats["status"] = "skipped (existing output)"
        return stats

    images = list_images(input_dir)
    stats["total"] = len(images)

    if not images:
        stats["status"] = "skipped (no input images)"
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                detection = mtcnn.detect(rgb_image)

                # mtcnn.detect may return (boxes, probs), a single array, or None
                if detection is None:
                    boxes = None
                elif isinstance(detection, tuple):
                    boxes = detection[0]
                else:
                    boxes = detection

                if boxes is None or len(boxes) != 1:
                    continue

                box = np.asarray(boxes[0], dtype=float)
                clipped = clip_box(box, rgb_image.width, rgb_image.height)
                if clipped is None:
                    continue

                cropped_face = rgb_image.crop(clipped)
                cropped_face.save(output_dir / image_path.name)
                stats["kept"] += 1
        except Exception as exc:
            print(f"  {season}/{character}: error processing {image_path.name}: {exc}")

    stats["status"] = "processed"
    return stats


def print_summary(results: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 72)
    print("MTCNN FACE EXTRACTION SUMMARY")
    print("=" * 72)
    print(f"{'Season':<8} | {'Character':<10} | {'Kept/Total':<11} | {'Retention':>9} | Status")
    print("-" * 72)

    grand_total = 0
    grand_kept = 0

    for item in results:
        total = int(item["total"])
        kept = int(item["kept"])
        status = str(item["status"])
        grand_total += total
        grand_kept += kept

        if total > 0:
            retention = f"{(kept / total) * 100:8.2f}%"
            kept_total = f"{kept}/{total}"
        else:
            retention = f"{'--':>8}"
            kept_total = "--"

        print(
            f"{item['season']:<8} | {item['character']:<10} | {kept_total:<11} | {retention:>9} | {status}"
        )

    print("-" * 72)
    if grand_total > 0:
        overall = (grand_kept / grand_total) * 100
        print(f"OVERALL   | {'':<10} | {grand_kept}/{grand_total:<11} | {overall:8.2f}% | done")
    else:
        print(f"OVERALL   | {'':<10} | {'--':<11} | {'--':>9} | done")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract single-face crops from dataset/S4 or dataset/S5 raw_frames using facenet_pytorch MTCNN."
    )
    parser.add_argument(
        "--season",
        choices=SEASONS,
        help="Process only one season. Omit to process both S4 and S5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = [args.season] if args.season else SEASONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)

    print("=" * 72)
    print("RAW FRAMES FACE DETECTION")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Characters: {', '.join(CHARACTERS)}")

    results: List[Dict[str, Any]] = []

    for season in seasons:
        print()
        print(f"Processing {season}...")
        for character in CHARACTERS:
            result = process_character(mtcnn, season, character)
            results.append(result)

            total = int(result["total"])
            kept = int(result["kept"])
            status = str(result["status"])

            if total > 0:
                print(f"  {character:<10}: {kept:4}/{total:<4} kept | {status}")
            else:
                print(f"  {character:<10}: --/-- kept | {status}")

    print_summary(results)


if __name__ == "__main__":
    main()