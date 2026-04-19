from unicodedata import category
import pandas as pd
import numpy as np


# load the csv
df=pd.read_csv("../data/Furniture.csv")
# print(df.head(5).to_string())
# print(df.columns)

# SELECT USEFI+ULL FEATURES
cols=['price','category','material','color','brand','season']
df = df[cols]
# print(df.head().to_string())


# CREATE STYLE
def get_style(price):
    if price>400:
        return "luxury"
    elif price>200:
        return "modern"
    else:
        return "minimalist"

df['style']=df['price'].apply(get_style)
# print("1. after adding style \n",df.head().to_string())


# CREATING THE DESCIPTION
def create_description(row):
    return (
        f"{row['style']} {row['material']} {row['category']} in {row['color']} color. "
        f"brand: {row['brand']}. season: {row['season']}. price: {round(row['price'],2)}"
    )

df['description']=df.apply(create_description,axis=1)
print("\n 2. after adding description \n",df['brand'].head().to_string())