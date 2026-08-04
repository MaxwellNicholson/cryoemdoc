# cryoEMdoc

`cryoEMdoc` is an inference-first Python package for cryo-EM image triage and documentation. It packages the prototype notebook behavior into reusable Python functions and a command line interface.

The current package supports:

- Step 1 image classification: `Atlas`, `Square`, `Protein`, or `Other`
- Square analyzer inference
- Atlas analyzer inference without CSV features
- Atlas prerecognition CSV generation
- Atlas analyzer inference with generated or user-provided CSV features

Protein-image analysis is not implemented yet. Protein images save a structured `not_implemented` response.

## Installation

From PyPI, once the first release is published:

```bash
pip install cryoemdoc
cryoemdoc download-models
```

If this doesn't work please try this:

```bash
py -m pip install git+https://github.com/MaxwellNicholson/cryoemdoc.git
cryoemdoc download-models
```

Local editable install:

```bash
pip install -e .
cryoemdoc download-models
```

For MRC support:

```bash
pip install -e ".[mrc]"
```

For tests:

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

```python
from cryoemdoc import analyze_image, classify_image, analyze_square, analyze_atlas

summary = analyze_image(
    "path/to/image.png",
    output="prediction.json",
    device="auto",
)
print(summary)
```

Prediction functions save standardized JSON and return a short printable summary with the rating, tags, and recommendations. If `output` is omitted or blank, the JSON file is saved in the current working directory.

`atlas_features` supports:

- `"none"`: run the atlas image-only analyzer; this is the default
- `"generate"`: run atlas prerecognition, write an atlas summary CSV, then run the atlas-with-CSV analyzer
- a CSV path: use an existing atlas feature table with the atlas-with-CSV analyzer

You can also use `use_atlas_csv=True` as a simple on switch:

```python
summary = analyze_image("path/to/atlas.png", use_atlas_csv=True)
print(summary)
```

## CLI

```bash
cryoemdoc download-models
cryoemdoc analyze image.png
cryoemdoc analyze image.png --use-atlas-csv
cryoemdoc classify image.png
cryoemdoc square image.png
cryoemdoc atlas image.png --atlas-features none
cryoemdoc atlas image.png --use-atlas-csv
cryoemdoc atlas-prerecognize atlas.png --output atlas_summary_scores.csv
```

Most commands also accept a folder path and recurse by default:

```bash
cryoemdoc classify ./images --output predictions.json
cryoemdoc atlas ./atlases --atlas-features none --output atlas_predictions.json
cryoemdoc analyze ./mixed_images --use-atlas-csv --output batch_results.json
```

## Python API

```python
from cryoemdoc import classify_image

summary = classify_image("image.png", device="cpu")
print(summary)
```

Folder/batch helpers save all per-image predictions in the standardized JSON file and return a summary that shows the first image:

```python
from cryoemdoc import analyze_images, classify_images, analyze_square_images, analyze_atlas_images

summary = analyze_images("./mixed_images", output="mixed_predictions.json")
summary_with_csv = analyze_images("./mixed_images", use_atlas_csv=True)
classification_summary = classify_images("./images")
square_summary = analyze_square_images("./squares")
atlas_summary = analyze_atlas_images("./atlases", use_atlas_csv=True)
```

For `analyze_images(..., use_atlas_csv=True)`, generated atlas summary CSV files are written into per-image subdirectories so batch runs do not overwrite earlier CSV outputs. For `analyze_atlas_images(..., use_atlas_csv=True)`, one combined `atlas_summary_scores.csv` is generated for the atlas folder and reused for all atlas images.

CSV mode now writes two files by default:

- `atlas_summary_scores.csv`: atlas-level square counts and `atlas_quality_score`
- `atlas_csv_analyzer_predictions.csv`: output from the trained atlas-with-CSV model, including predicted rating, tags, probabilities, and recommendations

To save the standardized output from Python, pass `output`. The lower-level atlas CSV analyzer prediction table can still be saved with `results_output`:

```python
summary = analyze_atlas_images(
    "./atlases",
    output="standardized_atlas_predictions.json",
    use_atlas_csv=True,
    results_output="atlas_predictions.csv",
)
```

From the CLI, use `--output` to choose the standardized JSON path:

```bash
cryoemdoc atlas ./atlases --use-atlas-csv --output atlas_predictions.json
```

```python
from cryoemdoc import analyze_square

summary = analyze_square("GridSquare_001.jpg")
print(summary)
```

```python
from cryoemdoc import analyze_atlas

atlas_summary = analyze_atlas("Atlas_001.jpg", atlas_features="none")
atlas_generated_csv_summary = analyze_atlas("Atlas_001.jpg", use_atlas_csv=True)
atlas_csv_summary = analyze_atlas("Atlas_001.jpg", atlas_features="atlas_summary_scores.csv")
```

```python
from cryoemdoc import atlas_prerecognize

csv_result = atlas_prerecognize(
    "Atlas_001.jpg",
    output="atlas_summary_scores.csv",
    save_annotated_images=True,
)
```

By default, `atlas_prerecognize` writes one row per atlas with the square counts and atlas quality score used by the CSV analyzer. To also inspect every detected square:

```bash
cryoemdoc atlas-prerecognize ./atlases --write-square-details
```

## Model Artifacts

The Python package includes small JSON metadata artifacts under `src/cryoemdoc/artifacts/`.
The `.pt` model weights are distributed separately as GitHub Release assets so the
PyPI package stays small.

- `image_classifier/best_resnet18_step1_classifier.pt`
- `square_analyzer/best_model_state.pt`
- `atlas_analyzer_without_csv/best_model_state.pt`
- `atlas_analyzer_with_csv/best_model_state.pt`
- saved `label_mappings.json`, `thresholds.json`, metadata, and tabular preprocessing JSON

Download the default model release:

```bash
cryoemdoc download-models
```

This downloads `cryoemdoc-models-v0.1.0.zip` from the `v0.1.0` GitHub Release
and installs it into `~/.cache/cryoemdoc/models/v0.1.0/`.

The package can also load artifacts from an external root:

```bash
export CRYOEMDOC_MODEL_DIR=/path/to/model_root
```

or:

```python
summary = analyze_square("image.jpg", model_dir="/path/to/model_root")
```

The external model root should contain the same subfolder layout as `src/cryoemdoc/artifacts`.

Maintainer instructions for building and uploading the model zip are in
`docs/github_releases.md`.

## Supported Inputs

The classifier supports PNG, JPEG, TIFF, and MRC-style grayscale inputs. Square and atlas analyzers use Pillow-readable image formats such as PNG, JPEG, TIFF, BMP, and WebP. Atlas prerecognition supports PNG, JPEG, TIFF, and MRC when optional readers are installed.

## Behavior Preserved From Notebooks

The package preserves the notebook inference choices:

- classifier class order: `Atlas`, `Square`, `Protein`, `Other`
- classifier 224 px square-pad preprocessing
- analyzer 384 px grayscale-to-3-channel preprocessing
- ResNet18 classifier and multitask analyzer architectures
- square and atlas tag/rating label mappings
- saved tag and rating thresholds
- square recommendation text
- atlas recommendation text
- atlas prerecognition ordered labels and CSV column names
- generated atlas CSV mode uses one atlas-level summary row per image by default
- atlas-with-CSV tabular feature order, fill values, means, and standard deviations

The classifier routing bug from the notebook is fixed by normalizing labels before route lookup.

## Limitations

- Protein analyzer inference is not implemented.
- The first public release is inference-first; training code remains documented but not packaged as a polished API.
- Full inference requires `torch`, `torchvision`, and the model weights.
- Atlas CSV generation requires `opencv-python-headless`.

## Citation / Acknowledgment

No formal citation is available yet. If you use this prototype in publications or shared workflows, acknowledge the cryoEMdoc project and include the repository URL once published.
