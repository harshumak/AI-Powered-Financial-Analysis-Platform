"""
Unit and Integration Tests for AI Financial Analysis Platform

USAGE:
    pytest tests/test_pipeline.py -v
"""

import pytest
import os
import sys
import shutil
import time
import pandas as pd
from pyspark.sql import SparkSession
import sqlite3
import gc  # Garbage collector

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from config.config import STOCK_DATA_DIR, PARQUET_PATH, DB_PATH
from data_collection.stock_downloader import StockDownloader
from preprocessing.spark_preprocessor import SparkPreprocessor
from sql_interface.database_manager import DatabaseManager
from ml_models.spark_gbt_forecaster import SparkGBTForecaster
from ml_models.investment_classifier import InvestmentClassifier

# --- FIXTURES (Setup & Teardown) ---

@pytest.fixture(scope="session")
def spark():
    """Create a single SparkSession for all tests"""
    spark = SparkSession.builder \
        .appName("TestSession") \
        .master("local[1]") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def clean_environment():
    """Clean up test data before and after tests (Windows Safe)"""
    test_dir = "tests/test_data"
    
    # --- SETUP ---
    # Try to clean up from previous runs, ignore errors if locked
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
        except PermissionError:
            pass # Ignore validation errors on setup
            
    os.makedirs(test_dir, exist_ok=True)
    
    yield test_dir
    
    # --- TEARDOWN ---
    # Force Garbage Collection to close any lingering DB connections
    gc.collect() 
    
    # Retry loop to handle Windows file locking lag
    if os.path.exists(test_dir):
        retries = 5
        for i in range(retries):
            try:
                shutil.rmtree(test_dir)
                break  # Success!
            except PermissionError:
                time.sleep(0.5)  # Wait 0.5s and try again
                if i == retries - 1:
                    print(f"⚠️ Warning: Could not delete {test_dir} due to file lock.")

# --- 1. DATA COLLECTION TESTS ---

def test_stock_downloader_single_ticker(clean_environment):
    """Test downloading data for a single ticker"""
    # Using a short date range for speed
    downloader = StockDownloader(tickers=["AAPL"], start_date="2023-01-01", end_date="2023-01-05")
    
    data_map = downloader.download_all()
    df = data_map.get("AAPL")
    
    assert df is not None
    assert not df.empty
    assert "Close" in df.columns
    assert len(df) > 0
    print("\n✅ Stock Downloader Test Passed")

def test_data_validation():
    """Test data validation logic"""
    downloader = StockDownloader()
    
    # Create valid dummy data
    valid_df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=250),
        'Open': [100] * 250, 'High': [105] * 250, 'Low': [95] * 250, 
        'Close': [102] * 250, 'Volume': [1000] * 250
    })
    
    # Create invalid dummy data (missing values)
    invalid_df = valid_df.copy()
    invalid_df.loc[0, 'Close'] = None
    
    assert downloader.validate_data({'VALID': valid_df}) == True
    # Should fail due to missing value
    assert downloader.validate_data({'INVALID': invalid_df}) == False
    print("\n✅ Validation Logic Test Passed")

# --- 2. FEATURE ENGINEERING TESTS ---

def test_spark_preprocessor(spark):
    """Test feature engineering calculations"""
    # Create dummy spark dataframe
    data = [("AAPL", "2023-01-01", 100.0), 
            ("AAPL", "2023-01-02", 102.0),
            ("AAPL", "2023-01-03", 104.0),
            ("AAPL", "2023-01-04", 101.0),
            ("AAPL", "2023-01-05", 99.0)]
    df = spark.createDataFrame(data, ["Ticker", "Date", "Close"])
    
    processor = SparkPreprocessor(spark)
    
    # Test Moving Average calculation
    df_ma = processor.calculate_moving_averages(df, windows=[2])
    assert "MA_2" in df_ma.columns
    
    # Test RSI calculation structure
    df_rsi = processor.calculate_rsi(df, window=2)
    assert "RSI" in df_rsi.columns
    
    print("\n✅ Spark Preprocessor Test Passed")

# --- 3. DATABASE TESTS ---

def test_database_manager(clean_environment):
    """Test SQLite creation and insertion"""
    test_db_path = os.path.join(clean_environment, "test.db")
    db = DatabaseManager(db_path=test_db_path)
    
    # 1. Test Create Tables
    db.create_tables()
    assert os.path.exists(test_db_path)
    
    # 2. Test Insert/Query
    conn = sqlite3.connect(test_db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO stock_data (ticker, date, close) VALUES ('TEST', '2023-01-01', 150.0)")
        conn.commit()
    finally:
        conn.close() # Ensure close happens even if error
    
    df = db.get_stock_data("TEST")
    assert len(df) == 1
    assert df.iloc[0]['close'] == 150.0
    
    # Explicitly clear internal references if any exist
    del db 
    print("\n✅ Database Manager Test Passed")

# --- 4. INTEGRATION TEST ---

def test_full_pipeline_flow(clean_environment, spark):
    """
    Integration Test: Runs the sequence of components to ensure they connect.
    """
    print("\n--- Running Integration Test ---")
    
    # 1. Download
    downloader = StockDownloader(tickers=["AAPL"], start_date="2023-11-01", end_date="2023-11-05")
    data_map = downloader.download_all()
    assert "AAPL" in data_map
    
    # 2. Preprocess
    pdf = data_map["AAPL"]
    sdf = spark.createDataFrame(pdf)
    processor = SparkPreprocessor(spark)
    sdf = processor.calculate_moving_averages(sdf, windows=[2])
    assert "MA_2" in sdf.columns
    
    # 3. Database (Mocked path)
    test_db = os.path.join(clean_environment, "integration.db")
    db = DatabaseManager(db_path=test_db)
    db.create_tables()
    
    # Clean up explicitly for Windows
    del db
    
    print("\n✅ Full Pipeline Integration Test Passed!")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))