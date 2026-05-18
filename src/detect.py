import sys
import subprocess
from pathlib import Path

def main():
    python = sys.executable
    script = Path(__file__).resolve().parent.parent / "extract_faces_mtcnn.py"
    args = [python, str(script)]
    # forward optional --season argument
    if "--season" in sys.argv:
        idx = sys.argv.index("--season")
        if idx + 1 < len(sys.argv):
            args += ["--season", sys.argv[idx + 1]]
    subprocess.run(args, check=True)

if __name__ == "__main__":
    main()
