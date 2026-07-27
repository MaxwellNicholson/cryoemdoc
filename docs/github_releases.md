# GitHub Release Model Artifacts

cryoEMdoc keeps source code in Git and publishes model weights as GitHub Release
assets. This keeps the Python package small while still giving users one command
to download the trained prototype models.

## Expected Asset Names

For release `v0.1.0`, upload these assets to the matching GitHub Release:

- `cryoemdoc-models-v0.1.0.zip`
- `cryoemdoc-models-v0.1.0.zip.sha256`

The zip must contain this root layout:

```text
cryoemdoc-models-v0.1.0/
  image_classifier/
    best_resnet18_step1_classifier.pt
  square_analyzer/
    best_model_state.pt
    best_model_metadata.json
    label_mappings.json
    thresholds.json
  atlas_analyzer_without_csv/
    best_model_state.pt
    best_model_metadata.json
    label_mappings.json
    thresholds.json
  atlas_analyzer_with_csv/
    best_model_state.pt
    best_model_metadata.json
    label_mappings.json
    thresholds.json
    tabular_preprocessing.json
```

## Build The Model Zip

Run this from a checkout that has the local `.pt` files:

```bash
python scripts/build_model_release.py --version v0.1.0
```

If the weights live somewhere else:

```bash
python scripts/build_model_release.py \
  --source-artifacts /path/to/artifacts \
  --output-dir dist_release \
  --version v0.1.0
```

The script writes:

```text
dist_release/cryoemdoc-models-v0.1.0.zip
dist_release/cryoemdoc-models-v0.1.0.zip.sha256
```

It does not move, delete, or rewrite the source artifacts.

## Create The Release

Create and push the release tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `Release` GitHub Actions workflow creates or updates the GitHub Release and
attaches the Python source/wheel distributions. Then upload the two model files
from `dist_release/` to that same release.

With GitHub CLI:

```bash
gh release upload v0.1.0 \
  dist_release/cryoemdoc-models-v0.1.0.zip \
  dist_release/cryoemdoc-models-v0.1.0.zip.sha256 \
  --clobber
```

Or in the GitHub web UI:

1. Open `https://github.com/MaxwellNicholson/cryoemdoc/releases/tag/v0.1.0`.
2. Click `Edit`.
3. Attach the zip and `.sha256` files.
4. Save the release.

## User Install Flow

After the release assets are attached, users can run:

```bash
pip install cryoemdoc
cryoemdoc download-models
cryoemdoc analyze path/to/image.png
```

`cryoemdoc download-models` downloads from:

```text
https://github.com/MaxwellNicholson/cryoemdoc/releases/download/v0.1.0/cryoemdoc-models-v0.1.0.zip
```

By default it installs models into:

```text
~/.cache/cryoemdoc/models/v0.1.0/
```

Users can override the model root at runtime:

```bash
cryoemdoc analyze image.png --model-dir /path/to/model_root
```

or with an environment variable:

```bash
export CRYOEMDOC_MODEL_DIR=/path/to/model_root
```
