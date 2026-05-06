import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

# Constants
NUM_PRODUCTS = 500
NUM_USERS = 2000

categories = ['Sofa', 'Chair', 'Table', 'Bed', 'Wardrobe']
materials = ['Wood', 'Metal', 'Glass', 'Fabric', 'Leather']
colors = ['Black', 'White', 'Grey', 'Brown', 'Beige', 'Blue']
styles = ['Modern', 'Contemporary', 'Vintage', 'Minimalist', 'Industrial']

# Ensure folders exist
os.makedirs("data/raw", exist_ok=True)

# -------------------------
# PRODUCT DATA GENERATION
# -------------------------
def generate_products():
    products = []

    for i in range(NUM_PRODUCTS):
        category = random.choice(categories)

        product = {
            "product_id": i,
            "name": f"{random.choice(styles)} {category}",
            "category": category,
            "material": random.choice(materials),
            "color": random.choice(colors),
            "style": random.choice(styles),
            "price": round(random.uniform(1000, 50000), 2),
            "rating": round(random.uniform(2.5, 5.0), 1),
        }

        # Description (important for embeddings later)
        product["description"] = f"{product['style']} {product['color']} {product['material']} {product['category']}"

        products.append(product)

    df = pd.DataFrame(products)
    df.to_csv("data/raw/products.csv", index=False)
    print("✅ Products generated")


# -------------------------
# USER BEHAVIOR GENERATION
# -------------------------
def generate_user_behavior():
    records = []

    for user_id in range(NUM_USERS):
        num_purchases = random.randint(1, 10)

        for _ in range(num_purchases):
            product_id = random.randint(0, NUM_PRODUCTS - 1)

            event_type = random.choice(["view", "cart", "purchase"])

            record = {
                "user_id": user_id,
                "product_id": product_id,
                "event_type": event_type,
                "timestamp": fake.date_time_between(start_date='-1y', end_date='now')
            }

            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv("data/raw/user_behavior.csv", index=False)
    print("✅ User behavior generated")


if __name__ == "__main__":
    generate_products()
    generate_user_behavior()