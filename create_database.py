#!/usr/bin/env python3
"""
Database Creation Script for Soccer Prediction Model

This script:
1. Checks if the database exists
2. Downloads Kaggle dataset if needed
3. Extracts and sets up the SQLite database
4. Verifies the database structure

Usage:
    python create_database.py [--force]
    
Options:
    --force    Force re-download even if database exists
"""

import os
import sys
import sqlite3
import zipfile
import argparse
from pathlib import Path

def check_kaggle_auth():
    """Check if Kaggle API is properly configured"""
    try:
        import kaggle
        # Test authentication by listing datasets
        kaggle.api.dataset_list(search="soccer", page_size=1)
        return True
    except Exception as e:
        print(f"Kaggle authentication failed: {e}")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Move downloaded kaggle.json to ~/.kaggle/")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_kaggle_dataset(data_dir="data", dataset="hugomathien/soccer"):
    """Download the Kaggle European Soccer dataset"""
    print(f"Downloading dataset: {dataset}")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Change to data directory for download
    original_dir = os.getcwd()
    os.chdir(data_dir)
    
    try:
        import kaggle
        kaggle.api.dataset_download_files(dataset, unzip=False)
        print("Dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False
    finally:
        os.chdir(original_dir)

def extract_dataset(data_dir="data", zip_filename="soccer.zip"):
    """Extract the downloaded dataset"""
    zip_path = os.path.join(data_dir, zip_filename)
    
    if not os.path.exists(zip_path):
        print(f"Zip file not found: {zip_path}")
        return False
    
    print(f"Extracting {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Dataset extracted successfully")
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def verify_database(db_path="data/database.sqlite"):
    """Verify the database structure and content"""
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return False
    
    print(f"Verifying database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['Match', 'Team', 'League', 'Country', 'Player', 'Player_Attributes', 'Team_Attributes']
        
        print(f"Found tables: {tables}")
        
        # Check if all expected tables exist
        missing_tables = [t for t in expected_tables if t not in tables]
        if missing_tables:
            print(f"Missing tables: {missing_tables}")
            return False
        
        # Check table sizes
        table_info = {}
        for table in expected_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_info[table] = count
            print(f"  {table}: {count:,} rows")
        
        # Basic sanity checks
        if table_info['Match'] < 20000:
            print("Warning: Match table has fewer rows than expected")
        
        if table_info['Team'] < 200:
            print("Warning: Team table has fewer rows than expected")
        
        conn.close()
        print("Database verification completed successfully")
        return True
        
    except Exception as e:
        print(f"Error verifying database: {e}")
        return False

def cleanup_files(data_dir="data"):
    """Clean up temporary files"""
    zip_path = os.path.join(data_dir, "soccer.zip")
    if os.path.exists(zip_path):
        print(f"Cleaning up {zip_path}")
        os.remove(zip_path)

def check_dependencies():
    """Check if required packages are installed"""
    missing_packages = []
    
    try:
        import kaggle
    except ImportError:
        missing_packages.append("kaggle")
    
    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Main function to set up the database"""
    parser = argparse.ArgumentParser(description='Create soccer prediction database')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-download even if database exists')
    parser.add_argument('--data-dir', default='data',
                       help='Directory for data files (default: data)')
    parser.add_argument('--keep-zip', action='store_true',
                       help='Keep the downloaded zip file')
    
    args = parser.parse_args()
    
    print("Soccer Database Creation Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Set up paths
    data_dir = args.data_dir
    db_path = os.path.join(data_dir, "database.sqlite")
    
    # Check if database already exists
    if os.path.exists(db_path) and not args.force:
        print(f"Database already exists: {db_path}")
        
        # Verify existing database
        if verify_database(db_path):
            print("Database is valid. Use --force to re-download.")
            return 0
        else:
            print("Database verification failed. Re-downloading...")
    
    # Check Kaggle authentication
    if not check_kaggle_auth():
        return 1
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset
    print(f"\nDownloading dataset to {data_dir}/")
    if not download_kaggle_dataset(data_dir):
        return 1
    
    # Extract dataset
    print("\nExtracting dataset...")
    if not extract_dataset(data_dir):
        return 1
    
    # Verify database
    print("\nVerifying database...")
    if not verify_database(db_path):
        return 1
    
    # Cleanup
    if not args.keep_zip:
        cleanup_files(data_dir)
    
    # Success message
    print("\n" + "=" * 50)
    print("DATABASE SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Database location: {os.path.abspath(db_path)}")
    print(f"Database size: {os.path.getsize(db_path) / (1024*1024):.1f} MB")
    
    # Show next steps
    print("\nNext steps:")
    print("1. Run: python explore_data.py")
    print("2. Run: python team_rating_model.py")
    print("3. Run: python predict_match.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())