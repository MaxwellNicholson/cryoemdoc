from __future__ import annotations

import sys

from cryoemdoc import analyze_square


if __name__ == "__main__":
    image_path = sys.argv[1]
    print(analyze_square(image_path))
