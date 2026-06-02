"""
Automation Script for Porter Delivery Time Estimation Data Preprocessing
Author: Dika Ramadani
Description: Automated preprocessing workflow for machine learning pipeline.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load_data(raw_data_path):
    """Load raw dataset from CSV file"""
    try:
        df = pd.read_csv(raw_data_path)
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {raw_data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)


def clean_data(df):
    """
    Clean the dataset:
    - Remove duplicates
    - Handle missing values in target
    - Handle logical anomalies (negative values)
    - Handle missing values safely
    """
    df_clean = df.copy()

    # 1. Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"✓ Duplicates removed: {duplicates_removed}")

    # 2. Drop baris yang targetnya (actual_delivery_time) kosong terlebih dahulu
    df_clean = df_clean.dropna(subset=["actual_delivery_time"])

    # 3. Filter nilai anomali negatif berdasarkan hasil statistik awal
    df_clean = df_clean[df_clean["min_item_price"] >= 0]
    for col in [
        "total_onshift_partners",
        "total_busy_partners",
        "total_outstanding_orders",
    ]:
        if col in df_clean.columns:
            df_clean = df_clean[(df_clean[col] >= 0) | (df_clean[col].isna())]
    print("✓ Logical anomalies (negative values) filtered")

    # 4. Drop fitur high cardinality 'store_id' agar tidak membebani encoding
    if "store_id" in df_clean.columns:
        df_clean = df_clean.drop(["store_id"], axis=1)
        print("✓ High cardinality feature 'store_id' dropped")

    # 5. Handle missing values di kolom kategorikal (object)
    categorical_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            df_clean[col] = df_clean[col].fillna("Unknown")
            print(
                f"✓ Handled {missing_count} missing values in '{col}' (filled with 'Unknown')"
            )

    # 6. Handle missing values di kolom numerik murni
    exclude_cols = [
        "market_id",
        "order_protocol",
        "created_at",
        "actual_delivery_time",
    ]
    numeric_cols = [
        c
        for c in df_clean.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    for col in numeric_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(
                f"✓ Handled {missing_count} missing values in '{col}' (filled with median)"
            )

    # Imputasi kelas modus untuk kolom kategori berbentuk angka
    for col in ["market_id", "order_protocol"]:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    print(f"✓ Data cleaning completed. Shape: {df_clean.shape}")
    return df_clean


def feature_engineering(df):
    """
    Feature Engineering:
    - Convert datetime columns
    - Extract time-based features
    - Calculate target variable (delivery_time_minutes)
    """
    df_eng = df.copy()

    # Konversi kolom waktu ke datetime
    df_eng["created_at"] = pd.to_datetime(df_eng["created_at"])
    df_eng["actual_delivery_time"] = pd.to_datetime(
        df_eng["actual_delivery_time"]
    )
    print("✓ DateTime columns converted")

    # Ambil durasi menit sebagai target variabel utama
    df_eng["delivery_time_minutes"] = (
        df_eng["actual_delivery_time"] - df_eng["created_at"]
    ).dt.total_seconds() / 60

    # Ekstrak fitur berbasis waktu (opsional untuk menambah variasi performa model kriteria 2)
    df_eng["order_hour"] = df_eng["created_at"].dt.hour
    df_eng["order_dayofweek"] = df_eng["created_at"].dt.dayofweek

    print("✓ Time-based features extracted")

    # Drop kolom datetime asli karena sudah diwakili oleh fitur durasi dan hour
    df_eng = df_eng.drop(["created_at", "actual_delivery_time"], axis=1)
    print(f"✓ Feature engineering completed. Shape: {df_eng.shape}")

    return df_eng


def encode_categorical(df):
    """
    Encode categorical variables:
    - Limit categories for store_primary_category
    - One-hot encoding for categorical features
    """
    df_encoded = df.copy()

    if "store_primary_category" in df_encoded.columns:
        top_20_categories = (
            df_encoded["store_primary_category"].value_counts().head(20).index
        )
        df_encoded["store_primary_category"] = df_encoded[
            "store_primary_category"
        ].apply(lambda x: x if x in top_20_categories else "Other")

    for col in ["market_id", "order_protocol"]:
        if col in df_encoded.columns:
            df_encoded[col] = (
                df_encoded[col]
                .fillna(-1)
                .astype(float)
                .astype(int)
                .astype(str)
                .replace("-1", "Unknown")
            )

    categorical_cols = df_encoded.select_dtypes(include=["object", "string"]).columns.tolist()

    if len(categorical_cols) > 0:
        df_encoded = pd.get_dummies(
            df_encoded, columns=categorical_cols, drop_first=True
        )
        print(
            f"✓ One-hot encoding applied to {len(categorical_cols)} categorical columns"
        )

        bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    print(f"✓ Encoding completed. Shape: {df_encoded.shape}")
    return df_encoded


def handle_outliers(df):
    """
    Handle outliers in delivery_time_minutes using business logic threshold
    """
    if "delivery_time_minutes" in df.columns:
        initial_count = len(df)
        # Batasi waktu pengiriman logis: di atas 0 menit dan maksimal 3 jam (180 menit)
        df = df[
            (df["delivery_time_minutes"] > 0)
            & (df["delivery_time_minutes"] <= 180)
        ].copy()
        outliers_removed = initial_count - len(df)

        print(f"✓ Outlier handling completed")
        print(f"  - Action: Filtered out delivery times <= 0 or > 180 minutes")
        print(f"  - Extreme rows removed: {outliers_removed}")
    else:
        print("⚠ Column 'delivery_time_minutes' not found for outlier handling")

    return df


def preprocess(raw_data_path, output_path):
    """Orchestrator pipeline function"""
    print("\n" + "=" * 60)
    print("STARTING RUNTIME AUTOMATION - ML DATA PREPROCESSING")
    print("=" * 60 + "\n")

    # Eksekusi pipeline secara bertahap
    df = load_data(raw_data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_categorical(df)
    df = handle_outliers(df)

    # Simpan dataset hasil pra-proses
    df.to_csv(output_path, index=False)
    print(f"\n✓ Preprocessed data saved successfully to: {output_path}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE SUCCESS")
    print("=" * 60 + "\n")

    return df


def main():
    """Main entry point configuration paths"""
    # Menentukan jalur relatif berdasarkan letak direktori kerja saat ini
    # Jalur diatur agar aman jika dijalankan dari root project maupun folder preprocessing
    base_dir = Path(__file__).resolve().parents[2]

    raw_data_path = (
        base_dir
        / "Eksperimen_SML_Dika-Ramadani"
        / "porter-delivery-time-estimation_raw"
        / "dataset.csv"
    )
    output_path = (
        base_dir
        / "Eksperimen_SML_Dika-Ramadani"
        / "preprocessing"
        / "porter-delivery-time-estimation_preprocessing"
        / "porter_delivery_preprocessed.csv"
    )

    # Buat direktori tujuan otomatis jika belum tersedia
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocess(str(raw_data_path), str(output_path))


# Blok eksekusi utama yang wajib ada di baris paling bawah skrip python
if __name__ == "__main__":
    main()