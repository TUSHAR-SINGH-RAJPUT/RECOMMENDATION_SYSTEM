import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# Ensure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
products = pd.read_csv("data/raw/products.csv")
behavior = pd.read_csv("data/raw/user_behavior.csv")

behavior['timestamp'] = pd.to_datetime(behavior['timestamp'])

# -------------------------
# CLEANING
# -------------------------

# Normalize text
products['name'] = products['name'].str.lower()
products['category'] = products['category'].str.lower()

# Handle missing (if any)
products.fillna({
    "price": products["price"].mean(),
    "rating": products["rating"].mean()
}, inplace=True)

# -------------------------
# ENCODING
# -------------------------
le_category = LabelEncoder()
le_material = LabelEncoder()
le_style = LabelEncoder()

products['category_encoded'] = le_category.fit_transform(products['category'])
products['material_encoded'] = le_material.fit_transform(products['material'])
products['style_encoded'] = le_style.fit_transform(products['style'])

# -------------------------
# PRODUCT FEATURES
# -------------------------

# Price category
def price_category(price):
    if price < 10000:
        return "budget"
    elif price < 30000:
        return "mid"
    else:
        return "premium"

products['price_category'] = products['price'].apply(price_category)

# Room type
def room_type(category):
    if category in ['sofa']:
        return 'living_room'
    elif category in ['bed', 'wardrobe']:
        return 'bedroom'
    else:
        return 'office'

products['room_type'] = products['category'].apply(room_type)

# Save product features
products.to_parquet("data/processed/products_clean.parquet")
print("✅ Product features saved")

# -------------------------
# RFM FEATURES (USER LEVEL)
# -------------------------

# Only purchases
purchases = behavior[behavior['event_type'] == 'purchase']

# Current time
now = datetime.now()

rfm = purchases.groupby('user_id').agg({
    'timestamp': lambda x: (now - x.max()).days,
    'product_id': 'count'
}).rename(columns={
    'timestamp': 'recency',
    'product_id': 'frequency'
})

# Monetary
merged = purchases.merge(products, on='product_id')

monetary = merged.groupby('user_id')['price'].sum()

rfm['monetary'] = monetary

rfm = rfm.fillna(0)

# Save user features
rfm.to_parquet("data/processed/user_features.parquet")
print("✅ User features saved")