from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine.benchmark_report import summarize

if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
