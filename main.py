import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"


def run_stage(script_name: str, extra_args=None, dry_run: bool = False):
    try:
        print(f"Starting: {script_name}")
        cmd = [sys.executable, str(SRC / script_name)]
        if extra_args:
            cmd += extra_args

        if dry_run:
            print("Dry run: would execute:", " ".join(cmd))
            print(f"Skipped: {script_name}\n")
            return

        subprocess.run(cmd, check=True)
        print(f"Finished: {script_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Stage {script_name} failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"Stage {script_name} failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Orchestrate full pipeline stages.")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--season", choices=["S4", "S5"], help="Season for --detect")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run consolidate -> split -> train -> evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")

    args = parser.parse_args()

    if args.all:
        # exclude detect
        run_stage("consolidate.py", dry_run=args.dry_run)
        run_stage("split.py", dry_run=args.dry_run)
        run_stage("train.py", dry_run=args.dry_run)
        run_stage("evaluate.py", dry_run=args.dry_run)
        return

    if args.consolidate:
        run_stage("consolidate.py", dry_run=args.dry_run)
    if args.split:
        run_stage("split.py", dry_run=args.dry_run)
    if args.detect:
        extra = []
        if args.season:
            extra = ["--season", args.season]
        run_stage("detect.py", extra_args=extra, dry_run=args.dry_run)
    if args.train:
        run_stage("train.py", dry_run=args.dry_run)
    if args.evaluate:
        run_stage("evaluate.py", dry_run=args.dry_run)

    # If no flags provided, print help
    if not any([args.consolidate, args.split, args.detect, args.train, args.evaluate, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
