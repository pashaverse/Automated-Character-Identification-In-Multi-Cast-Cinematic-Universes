import sys
import subprocess
from pathlib import Path

def main():
    python = sys.executable
    script = Path(__file__).resolve().parent.parent / "train_resnet50.py"
    subprocess.run([python, str(script)], check=True)

if __name__ == "__main__":
    main()
