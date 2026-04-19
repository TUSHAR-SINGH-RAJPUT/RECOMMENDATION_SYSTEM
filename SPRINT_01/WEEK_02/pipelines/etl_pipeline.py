import pandas as pd
import os
import logging
from datetime import datetime

# -----------------------------
# Setup logging configuration
# -----------------------------
# filename → log file where all logs will be stored
# level → minimum level of logs to capture (INFO, ERROR, etc.)
# format:
#   %(asctime)s  → timestamp of log
#   %(levelname)s → type of log (INFO, ERROR, WARNING)
#   %(message)s   → actual log message
logging.basicConfig(
    filename="etl_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# File paths configuration
# -----------------------------
# BASE_DIR → directory where this script is located
# INPUT_FILE → path to raw dataset (one level above pipelines folder)
# OUTPUT_FILE → path to save processed dataset
BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.abspath(os.path.join(BASE_DIR, "..","data" ,"user_behavior.csv"))
OUTPUT_FILE = output_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "data", "processed_user_behavior.csv")
    )

# -----------------------------
# Extract Phase
# -----------------------------
# Reads raw data from CSV file
# Returns a pandas DataFrame
def extract():
    try:
        df = pd.read_csv(INPUT_FILE)
        logging.info("Data extracted successfully")
        return df
    except Exception as e:
        logging.error(f"Error in extract: {e}")
        raise

# -----------------------------
# Transform Phase
# -----------------------------
# Cleans and processes raw data:
# - Removes whitespace from column names
# - Drops missing values
# - Converts data types
# - Maps interaction types to numeric scores
def transform(df):
    try:
        # Clean column names
        df.columns = df.columns.str.strip()

        # Remove rows with missing values
        df = df.dropna()

        # Convert user_id and product_id to integers
        df["user_id"] = df["user_id"].astype(int)
        df["product_id"] = df["product_id"].astype(int)

        # Define interaction weights for scoring
        interaction_weights = {
            "view": 1,
            "cart": 2,
            "purchase": 3
        }

        # Map interaction types to numerical scores
        df["interaction_score"] = df["interaction_type"].map(interaction_weights)

        # Remove rows where mapping failed (invalid interaction types)
        df = df.dropna(subset=["interaction_score"])

        logging.info("Data transformed successfully")
        return df

    except Exception as e:
        logging.error(f"Error in transform: {e}")
        raise

# -----------------------------
# Load Phase
# -----------------------------
# Saves the processed DataFrame to a CSV file
def load(df):
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Data loaded successfully")
    except Exception as e:
        logging.error(f"Error in load: {e}")
        raise

# -----------------------------
# Pipeline Runner
# -----------------------------
# Executes the full ETL pipeline:
# Extract → Transform → Load
def run_pipeline():
    logging.info("ETL Pipeline Started")

    df = extract()
    df = transform(df)
    load(df)

    logging.info("ETL Pipeline Completed Successfully")
    print("✅ ETL Pipeline Completed")

# -----------------------------
# Entry Point
# -----------------------------
# Ensures the pipeline runs only when the script is executed directly
if __name__ == "__main__":
    run_pipeline()