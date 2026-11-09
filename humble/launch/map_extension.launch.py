from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from n3_launch import describe


def generate_launch_description():
    return describe("map_extension")
