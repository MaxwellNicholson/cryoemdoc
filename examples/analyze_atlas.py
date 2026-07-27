from __future__ import annotations

import sys

from cryoemdoc import analyze_atlas


if __name__ == "__main__":
    image_path = sys.argv[1]
    atlas_features = sys.argv[2] if len(sys.argv) > 2 else "none"
    print(analyze_atlas(image_path, atlas_features=atlas_features))
