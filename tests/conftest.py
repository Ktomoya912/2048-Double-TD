import sys
from pathlib import Path

# Set up sys.argv before importing anything from src that uses argparse
# This prevents argparse errors when common.args.parse_args() is called at import time
if len(sys.argv) == 1 or not any(arg in sys.argv for arg in ["-m", "--model"]):
    # If running via pytest, mock the required arguments
    sys.argv = ["pytest", "-m", "CNN_BASE", "--trainer", "TDA"]

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
