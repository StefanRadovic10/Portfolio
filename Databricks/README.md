# Fare Prediction — Databricks & PySpark

Machine learning project built on Databricks for predicting ride fare amounts using a dataset of 38M+ records stored in Azure Data Lake Storage (Parquet format).

## Tech Stack

Databricks, Apache Spark, PySpark, Spark SQL, PySpark MLlib, PyTorch, Azure ADLS Gen2

## What's in this project

**Data Cleaning** — dropping rows with missing critical columns, removing duplicates, filling logical zeros for fee-related columns, dropping rows missing core ML fields.

**Outlier Removal** — filtering out physically impossible values: zero or extreme distances, negative durations, invalid passenger counts, fares below the minimum threshold.

**Feature Engineering** — trip duration in minutes derived from timestamps, pickup hour and day of week extracted to capture rush hours and weekly patterns.

**Encoding** — location and vendor ID columns encoded via StringIndexer (fitted on train only to prevent data leakage), all features assembled into a single vector column.

**Spark SQL Analysis** — exploratory queries covering trip counts per day, average fare and distance, revenue by hour, average trip duration and top pickup locations.

**Train / Validation / Test Split** — data split into 64% train, 16% validation and 20% test. A 5% sample (~2M records) was used during model selection for faster iteration, with the best model retrained on the full dataset.

**Models**
- Linear Regression (baseline)
- Random Forest — 20 trees, max depth 8, R² ≈ 0.96, MAE ≈ $1
- Gradient Boosting (GBT) — sequential boosting optimized for speed and accuracy

**Neural Network** — custom feed-forward regression network (FareNet) built in PyTorch, 3 fully connected layers with ReLU activations, trained on data converted from Spark DataFrames into tensors with full cleaning and outlier filtering applied.

## Results

Random Forest achieved the best results: average error around $1 for cheaper rides, ~$3.5 for expensive ones, explaining 96% of fare variance. Most important features were trip distance and duration, followed by pickup and dropoff location IDs.
