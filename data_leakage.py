import pandas as pd
import logging 
import sys
import os
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# =============================================================================
# 0. LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),           
        logging.FileHandler("benchmark_execution.log") 
    ]
)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
logging.info("=" * 80)
logging.info("1. LOADING DATASETS")
logging.info("=" * 80)

COMP_ID = "2-el-1730-machine-learning-project-2026"

def find_data_path(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if "train.geojson" in files:
            return root
    return None

if os.path.exists('/kaggle/input'):
    DATA_DIR = find_data_path('/kaggle/input')
else:
    DATA_DIR = "./data"

if DATA_DIR is None:
    logging.critical("Impossible to locate the dataset. Please ensure 'train.geojson' and 'test.geojson' are in the correct directory.")
    sys.exit(1)


logging.info(f"Files inside: {DATA_DIR}")
# Local paths assuming files are in the same folder
try:
    train_df = gpd.read_file(os.path.join(DATA_DIR, "train.geojson"))
    test_df = gpd.read_file(os.path.join(DATA_DIR, "test.geojson"))
    logging.info(f"Train shape: {train_df.shape}")
    logging.info(f"Test shape:  {test_df.shape}")
except Exception as e:
    logging.critical("Failed to load datasets.", exc_info=True)
    sys.exit(1)


y = train_df['change_type']

# 2.  We prepare a "suspects" dataframe with only the 4 columns that could potentially leak information about the target variable.
suspects = pd.DataFrame()
suspects['x'] = train_df.geometry.centroid.x
suspects['y'] = train_df.geometry.centroid.y
suspects['urban'] = train_df['urban_type'].astype('category').cat.codes
suspects['geo'] = train_df['geography_type'].astype('category').cat.codes

# 3. Quick and dirty train/test split on these 4 columns suspects
X_train_hack, X_val_hack, y_train_hack, y_val_hack = train_test_split(suspects, y, test_size=0.2, random_state=42)

# 4. We launch a Random Forest on these 4 columns to see if we can predict the target variable. If we can, it means that these columns are leaking information about the target variable.
rf_hack = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)
rf_hack.fit(X_train_hack, y_train_hack)

# 5. We evaluate the performance of the model on the validation set. If we get a good performance, it means that these 4 columns are leaking information about the target variable and that we should not use them in our final model.
preds = rf_hack.predict(X_val_hack)
print(classification_report(y_val_hack, preds))

# 6. We check the feature importance to see which of the 4 columns is the most important for predicting the target variable. If one of them is much more important than the others, it means that it is the one that is leaking the most information about the target variable.
importances = pd.Series(rf_hack.feature_importances_, index=suspects.columns)
print("\n--- FEATURE IMPORTANCE ---")
print(importances.sort_values(ascending=False))