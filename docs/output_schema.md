# Output Schema

Prediction functions save standardized JSON files and return short printable
summary strings.

```python
from cryoemdoc import analyze_image

summary = analyze_image("image.png", output="prediction.json")
print(summary)
```

If `output` is omitted, `None`, or blank, the JSON file is saved in the current
working directory with a generated name such as
`image_standardized_prediction.json`.

## Single-Image Prediction

`classify_image`, `analyze_square`, `analyze_atlas`,
`analyze_atlas_without_csv`, `analyze_atlas_with_csv`, and `analyze_image`
save this shape:

```json
{
  "schema_version": "1.0",
  "task": "analyze_square",
  "image_path": "square.jpg",
  "analyzer": "square",
  "status": "ok",
  "predictions": {
    "image_type": {
      "label": "Square",
      "confidence": 0.98,
      "probabilities": {
        "Atlas": 0.01,
        "Square": 0.98,
        "Protein": 0.0,
        "Other": 0.01
      }
    },
    "tags": {
      "labels": ["no issues"],
      "probabilities": {
        "thick ice": 0.1,
        "non-uniform ice": 0.2,
        "ice contamination": 0.1
      },
      "thresholds": {
        "thick ice": 0.5
      }
    },
    "rating": {
      "label": "acceptable",
      "probabilities": {
        "good": 0.2,
        "acceptable": 0.7,
        "unacceptable": 0.1
      },
      "threshold": 0.5
    }
  },
  "recommendations": ["no recommendation"],
  "artifacts": {
    "standardized_output": "prediction.json"
  },
  "warnings": [],
  "errors": []
}
```

When a model has no predicted issue tags, the standardized tag label is always
`["no issues"]`. Classifier-only output uses the same shape but may have
`null` for rating fields and empty tag probabilities.

The returned summary string looks like:

```text
Saved standardized prediction to prediction.json.
Image: square.jpg
Image type: Square
Predicted rating: acceptable
Predicted tags: no issues
```

## Batch Prediction

`classify_images`, `analyze_square_images`, `analyze_atlas_images`, and
`analyze_images` save a batch object with one standardized item per image:

```json
{
  "schema_version": "1.0",
  "task": "analyze_images",
  "status": "ok",
  "input": {
    "path": "images",
    "type": "folder"
  },
  "items": [
    {
      "schema_version": "1.0",
      "task": "analyze_images",
      "image_path": "first.png",
      "analyzer": "atlas_without_csv",
      "status": "ok",
      "predictions": {
        "image_type": {
          "label": "Atlas",
          "confidence": 0.97,
          "probabilities": {}
        },
        "tags": {
          "labels": ["cracks"],
          "probabilities": {},
          "thresholds": {}
        },
        "rating": {
          "label": "unacceptable",
          "probabilities": {},
          "threshold": 0.5
        }
      },
      "recommendations": ["decrease glow discharge time/current"],
      "artifacts": {},
      "warnings": [],
      "errors": []
    }
  ],
  "artifacts": {
    "standardized_output": "images_standardized_predictions.json"
  },
  "warnings": [],
  "errors": []
}
```

The returned batch summary shows the first image only:

```text
Saved standardized predictions to images_standardized_predictions.json.
First image: first.png
Image type: Atlas
Predicted rating: unacceptable
Predicted tags: cracks
Showing 1 of 2 results; the rest are in the saved file.
```

## Atlas CSV Features

Atlas CSV mode is off by default. Pass `use_atlas_csv=True` or
`atlas_features="generate"` to generate atlas-level prerecognition CSV features
and route atlas images through the CSV analyzer.

Generated atlas summary CSVs contain one row per atlas:

```json
{
  "image_name": "atlas.png",
  "visible_square_count": 356,
  "good_square_count": 61,
  "non_uniform_square_count": 3,
  "cracked_square_count": 0,
  "bad_size_square_count": 202,
  "atlas_quality_score": 16.5,
  "error": ""
}
```

When atlas CSV mode generates prerecognition features, the standardized
`artifacts` object includes generated paths such as
`atlas_quality_scores_csv`, `atlas_csv_analyzer_predictions_csv`, and
`atlas_prerecognition`.

## Unsupported Types

Protein images currently save a standardized result with:

```json
{
  "status": "not_implemented",
  "predictions": {
    "image_type": {
      "label": "Protein"
    },
    "tags": {
      "labels": ["no issues"]
    },
    "rating": {
      "label": null
    }
  },
  "errors": ["Protein analyzer is not implemented yet."]
}
```
