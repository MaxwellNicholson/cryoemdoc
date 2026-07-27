from __future__ import annotations

import sys

from cryoemdoc import analyze_image


if __name__ == "__main__":
    image_path = sys.argv[1]
    print(analyze_image(image_path))
