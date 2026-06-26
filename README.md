# uvi-proj

## Predicting Socioeconomic Wealth from Street View Imagery

Estimating DHS wealth index scores from Google Street View imagery using computer vision features and regression, with the goal of scalable, low-cost poverty mapping across Sub-Saharan Africa.

## Overview

Ground-truth socioeconomic data in low- and middle-income countries is expensive and infrequently collected. This project explores whether passively available street-level imagery can serve as a proxy for household wealth, using DHS (Demographic and Health Surveys) wealth index scores as labels.

The pipeline extracts visual features from Google Street View images across Sub-Saharan African cities, trains regression models to predict cluster-level wealth scores, and evaluates generalization to other nearby regions. 

## Methods

### Data

Google Street View images sampled from DHS cluster locations in Nigeria and Kenya. Working on expansion to Senegal, Ghana, and Rwanda.
Labels: DHS wealth index scores aggregated to the cluster level
Images downloaded via the GSV Static API

### Feature extraction

CLIP (ViT-B/32) embeddings — captures semantic visual context
ResNet50 features — captures lower-level visual structure

### Modeling

Ridge regression with alpha sweep over a log-spaced grid
PCA applied to reduce dimensionality prior to regression
Evaluation: R² on a held-out test set


## Results

| Features | Model | R² |
|----------|-------|----:|
| ResNet50 | Ridge | 0.394 |
| ResNet50 | Ridge + PCA + Alpha Sweep | 0.480 |
| CLIP | Ridge | 0.463 |
| CLIP | Ridge + PCA + Alpha Sweep | 0.503 |


## Current status & next steps

This project is actively ongoing. Current steps include:
- Cross-country generalization: collecting GSV imagery and DHS data for Kenya, Rwanda, Ghana, and Senegal to test transfer learning
- Multimodal extensions: planning to incorporate VIIRS nighttime lights as an additional feature, and potentially semantic segmentation

The goal is to assess whether models trained on Nigeria transfer to other Sub-Saharan African contexts without retraining


## Notes

Raw GSV images and most data files are not committed to this repo due to size. 
