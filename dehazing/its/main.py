"""Small command entry point. Run from this folder."""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("train", "eval"))
    known, rest = p.parse_known_args()
    script = "train.py" if known.mode == "train" else "eval.py"
    subprocess.run([sys.executable, script, *rest], check=True)


if __name__ == "__main__":
    main()
