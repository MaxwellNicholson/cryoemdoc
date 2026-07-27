# Model Card

## Overview

cryoEMdoc currently includes four prototype inference artifacts:

- Step 1 ResNet18 image classifier
- Square ResNet18 multitask analyzer
- Atlas ResNet18 multitask analyzer without CSV features
- Atlas ResNet18 multitask analyzer with atlas prerecognition CSV features

These models were extracted from the prototype notebooks and packaged for inference. They should be treated as research prototypes until validated on external data.

## Inputs

The models expect cryo-EM-related image screenshots or microscopy-derived images. Classifier inputs are normalized to grayscale, duplicated into RGB, square-padded, resized to 224 px, and ImageNet-normalized. Square and atlas analyzers open images as grayscale, resize directly to 384 px, duplicate to RGB, and apply ImageNet normalization.

## Outputs

The classifier predicts one of:

- `Atlas`
- `Square`
- `Protein`
- `Other`

The square analyzer predicts issue tags and a rating:

- tags: `thick ice`, `non-uniform ice`, `ice contamination`
- ratings: `good`, `acceptable`, `unacceptable`

The atlas analyzers predict:

- tags: `cracks`, `non-uniform ice`, `thick ice`
- ratings: `acceptable`, `unacceptable`

## Thresholds

Inference uses the saved `thresholds.json` files from the prototype output directories. Tag predictions use per-tag thresholds. Square rating prediction gates `unacceptable` by threshold, then chooses `good` versus `acceptable` by argmax. Atlas rating prediction gates the binary `unacceptable` class by threshold.

## Intended Use

The intended use is triage and documentation assistance for cryo-EM screening workflows. The outputs should support expert review, not replace expert judgment.

## Limitations

- Protein analysis is not implemented.
- Validation was prototype-local; external validation is recommended before operational use.
- Atlas prerecognition is heuristic computer vision and may be sensitive to image style, contrast, annotations, and atlas layout.
- The model weights are large enough that public hosting should use Git LFS, GitHub Releases, or Hugging Face Hub.
