# Data Dictionary

## Overview

This project consists of two datasets:

1. **products.csv** – Contains product-related information such as category, price, and ratings.
2. **user_behavior.csv** – Contains user interaction data with products including views, cart actions, and purchases.

These datasets are designed for data analysis, data cleaning, and machine learning tasks such as recommendation systems.

---
# RAW DATASET

# 📦 Dataset 1: products.csv

## Description

This dataset contains details about products available in an e-commerce system.

## Columns

| Column Name  | Data Type | Description                        | Example                       | Notes                       |
| ------------ | --------- | ---------------------------------- | ----------------------------- | --------------------------- |
| product_id   | integer   | Unique identifier for each product | 101                           | Primary key                 |
| name         | string    | Name of the product                | Modern Sofa                   | May contain repeated names  |
| category     | string    | Product category                   | Furniture                     |                             |
| sub_category | string    | Sub-category of product            | Sofa                          |                             |
| price        | float     | Price of the product               | 4999.99                       |                             |
| brand        | string    | Brand name                         | IKEA                          |                             |
| material     | string    | Material used                      | Wood                          |                             |
| rating       | float     | Product rating (1–5 scale)         | 4.3                           | May contain low/high values |
| description  | string    | Product description                | Contemporary White Glass Sofa | Generated text              |

---

## Data Quality Notes

* Ratings are within a range of **1 to 5**
* Product names may not be unique
* Descriptions are synthetic and may not always match perfectly with attributes
* Possible inconsistencies in category vs description

---

# 👤 Dataset 2: user_behavior.csv

## Description

This dataset captures user interactions with products over time.

## Columns

| Column Name | Data Type | Description                       | Example                | Notes                  |
| ----------- | --------- | --------------------------------- | ---------------------- | ---------------------- |
| user_id     | integer   | Unique user identifier            | 1001                   |                        |
| product_id  | integer   | ID of the product interacted with | 101                    | Foreign key → products |
| event_type  | string    | Type of interaction               | view / cart / purchase |                        |
| timestamp   | datetime  | Time of interaction               | 2025-07-23 13:05:30    |                        |

---

## Data Quality Notes

* Multiple interactions per user are possible
* Same user can interact with the same product multiple times
* Event types include:

  * **view** → user viewed product
  * **cart** → added to cart
  * **purchase** → completed purchase
* Timestamp format is consistent but may require parsing

---

# 🔗 Relationship Between Datasets

* `product_id` in **user_behavior.csv** references `product_id` in **products.csv**
* This forms a **one-to-many relationship**:

  * One product → many user interactions

---

# ⚠️ Potential Data Issues

* Duplicate user interactions
* Imbalanced event types (more views than purchases)
* No user demographic information available
* Synthetic data may not reflect real-world distributions

---

# 🎯 Intended Use

* Data Cleaning Practice
* Exploratory Data Analysis (EDA)
* Recommendation System Development
* User Behavior Analysis
* Machine Learning Model Training

---


# CLEAN DATASET

# Data Dictionary (Processed Data)

## Overview

This project contains cleaned and processed datasets derived from raw e-commerce data. The data has been preprocessed to remove inconsistencies, handle missing values, and engineer useful features for analysis and machine learning.

Datasets included:

1. **products_clean.parquet** – Cleaned product-level data
2. **user_features.parquet** – Aggregated user behavior features

---

# 📦 Dataset 1: products_clean.parquet

## Description

This dataset contains cleaned and standardized product information. All inconsistencies from the raw dataset have been addressed.

## Columns

| Column Name  | Data Type | Description                        | Example           | Notes             |
| ------------ | --------- | ---------------------------------- | ----------------- | ----------------- |
| product_id   | integer   | Unique identifier for each product | 101               | Primary key       |
| name         | string    | Standardized product name          | Modern Sofa       | Cleaned text      |
| category     | string    | Product category                   | Furniture         | Normalized values |
| sub_category | string    | Product sub-category               | Sofa              |                   |
| price        | float     | Product price                      | 4999.99           | No missing values |
| brand        | string    | Brand name                         | IKEA              | Standardized      |
| material     | string    | Product material                   | Wood              | Cleaned values    |
| rating       | float     | Product rating (1–5 scale)         | 4.3               | Outliers handled  |
| description  | string    | Product description                | Contemporary sofa | Cleaned text      |

---

## Data Processing Performed

* Removed duplicate products
* Standardized text fields (name, brand, category)
* Handled missing values
* Ensured rating is within valid range (1–5)
* Cleaned inconsistent descriptions

---

# 👤 Dataset 2: user_features.parquet

## Description

This dataset contains aggregated user-level features derived from raw interaction data. It is structured for machine learning and behavioral analysis.

## Columns

| Column Name        | Data Type | Description                         | Example | Notes             |
| ------------------ | --------- | ----------------------------------- | ------- | ----------------- |
| user_id            | integer   | Unique user identifier              | 1001    | Primary key       |
| total_views        | integer   | Total number of product views       | 25      |                   |
| total_cart         | integer   | Total cart additions                | 5       |                   |
| total_purchases    | integer   | Total purchases made                | 2       |                   |
| avg_price_viewed   | float     | Average price of viewed products    | 3500.50 |                   |
| avg_price_cart     | float     | Average price of cart products      | 4200.00 |                   |
| avg_price_purchase | float     | Average price of purchased products | 5000.00 |                   |
| conversion_rate    | float     | Purchase conversion rate            | 0.08    | purchases / views |

---

## Feature Engineering Details

* Aggregated user interactions (view, cart, purchase)
* Calculated average price metrics for each interaction type
* Derived **conversion rate**:

  * purchases ÷ views
* Removed duplicate interactions
* Handled missing or null values
* Ensured numerical consistency across features

---

# 🔗 Relationship Between Datasets

* `product_id` connects product data to user interactions (in raw stage)
* Processed data is:

  * **products_clean** → product-level
  * **user_features** → user-level aggregated

---

# ⚠️ Notes

* Data is fully cleaned and ready for analysis
* No major missing values present
* Features are engineered for ML use cases
* Suitable for recommendation systems and user segmentation

---

# 🎯 Intended Use

* Machine Learning Model Training
* Recommendation Systems
* User Segmentation
* Behavioral Analytics
* Feature Engineering Practice

---

# 📌 Data Source

Kaaggle , Faker 
