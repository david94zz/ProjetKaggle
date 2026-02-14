# =============================================================================
# EXHAUSTIVE BENCHMARK - KAGGLE COMPETITION
# Geographical zone classification (6 classes)
# Metric: Mean F1-Score (macro)
# =============================================================================

import logging
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import warnings
from scipy import stats

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV

# Gradient Boosting Libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

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

# Local paths assuming files are in the same folder
try:
    train_df = gpd.read_file("train.geojson")
    test_df = gpd.read_file("test.geojson")
    logging.info(f"Train shape: {train_df.shape}")
    logging.info(f"Test shape:  {test_df.shape}")
except Exception as e:
    logging.critical("Failed to load datasets. Please check file paths.", exc_info=True)
    sys.exit(1)

TARGET_MAP = {
    'Demolition': 0, 'Road': 1, 'Residential': 2,
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}

# =============================================================================
# 2. MASSIVE FEATURE ENGINEERING
# =============================================================================
logging.info("\n" + "=" * 80)
logging.info("2. FEATURE ENGINEERING")
logging.info("=" * 80)

def extract_advanced_features(df):
    """
    Extracts an exhaustive list of features from images, geometry, and temporal data.
    """
    feat = pd.DataFrame(index=df.index)

    # A. Raw Image Features & Derived Color Indices
    for i in range(1, 6):
        rm = f'img_red_mean_date{i}'
        gm = f'img_green_mean_date{i}'
        bm = f'img_blue_mean_date{i}'
        rs = f'img_red_std_date{i}'
        gs = f'img_green_std_date{i}'
        bs = f'img_blue_std_date{i}'

        if rm in df.columns and gm in df.columns and bm in df.columns:
            r = pd.to_numeric(df[rm], errors='coerce').fillna(0)
            g = pd.to_numeric(df[gm], errors='coerce').fillna(0)
            b = pd.to_numeric(df[bm], errors='coerce').fillna(0)
            
            total = r + g + b + 1e-8
            
            # Brightness and Ratios
            feat[f'brightness_d{i}'] = total / 3.0
            feat[f'r_ratio_d{i}'] = r / total
            feat[f'g_ratio_d{i}'] = g / total
            feat[f'b_ratio_d{i}'] = b / total
            
            # Pseudo Vegetation and Water Indices (NDVI / NDWI approximations)
            feat[f'ndvi_like_d{i}'] = (g - r) / (g + r + 1e-8)
            feat[f'ndwi_like_d{i}'] = (g - b) / (g + b + 1e-8)

        # Standard Deviation features (Texture proxies)
        if rs in df.columns and gs in df.columns and bs in df.columns:
            feat[f'texture_mean_d{i}'] = (df[rs] + df[gs] + df[bs]) / 3.0

    # B. Temporal Image Differences (Date 5 vs Date 1)
    for color in ['red', 'green', 'blue']:
        c1 = f'img_{color}_mean_date1'
        c5 = f'img_{color}_mean_date5'
        if c1 in df.columns and c5 in df.columns:
            feat[f'total_diff_{color}'] = df[c5] - df[c1]
            feat[f'total_abs_diff_{color}'] = np.abs(df[c5] - df[c1])

    if 'brightness_d1' in feat.columns and 'brightness_d5' in feat.columns:
        feat['total_bright_diff'] = feat['brightness_d5'] - feat['brightness_d1']

    # C. Geometric Features
    if 'geometry' in df.columns:
        # Project geometry to metric system for accurate area/perimeter
        projected_geom = df.geometry.to_crs(epsg=3857)
        feat['area'] = projected_geom.area
        feat['perimeter'] = projected_geom.length
        feat['compactness'] = 4 * np.pi * feat['area'] / (feat['perimeter']**2 + 1e-8)
        
        # Interaction between size and final brightness
        if 'brightness_d5' in feat.columns:
            feat['area_x_bright5'] = feat['area'] * feat['brightness_d5']

    # D. Categorical Status Encoding
    status_cols = [f'change_status_date{i}' for i in range(5)]
    existing_status = [c for c in status_cols if c in df.columns]

    if existing_status:
        sv = df[existing_status].astype(str).values
        # Count number of status changes over time
        feat['n_status_changes'] = np.sum(sv[:, 1:] != sv[:, :-1], axis=1)
        feat['n_unique_status'] = pd.DataFrame(sv).nunique(axis=1).values

    # E. Multi-label Encoding for Urban and Geography Types
    if 'urban_type' in df.columns:
        urban_split = df['urban_type'].fillna('').astype(str).apply(
            lambda x: [s.strip().lower() for s in x.split(',') if s.strip()]
        )
        mlb_u = MultiLabelBinarizer()
        u_df = pd.DataFrame(mlb_u.fit_transform(urban_split), columns=[f'urb_{c}' for c in mlb_u.classes_], index=feat.index)
        feat = pd.concat([feat, u_df], axis=1)

    if 'geography_type' in df.columns:
        geo_split = df['geography_type'].fillna('').astype(str).apply(
            lambda x: [s.strip().lower() for s in x.split(',') if s.strip()]
        )
        mlb_g = MultiLabelBinarizer()
        g_df = pd.DataFrame(mlb_g.fit_transform(geo_split), columns=[f'geo_{c}' for c in mlb_g.classes_], index=feat.index)
        feat = pd.concat([feat, g_df], axis=1)

    return feat

try:
    logging.info("Extracting features for Train...")
    train_feat = extract_advanced_features(train_df)
    
    logging.info("Extracting features for Test...")
    test_feat = extract_advanced_features(test_df)

    common_cols = sorted(set(train_feat.columns) & set(test_feat.columns))
    X_train = train_feat[common_cols].values.astype(np.float64)
    X_test = test_feat[common_cols].values.astype(np.float64)
    y_train = train_df['change_type'].map(TARGET_MAP).values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    logging.info(f"Final X_train shape: {X_train.shape}")
    
    # Calculate custom aggressive class weights to save minority classes
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    custom_weights = {c: w for c, w in zip(classes, weights)}
    custom_weights[4] *= 2.0  # Industrial
    custom_weights[5] *= 10.0 # Mega Projects
    logging.info("Computed custom class weights to handle severe class imbalance.")

except Exception as e:
    logging.error("Error occurred during Feature Engineering phase.", exc_info=True)
    sys.exit(1)

# =============================================================================
# 3. MODEL BENCHMARKING
# =============================================================================
logging.info("\n" + "=" * 80)
logging.info("3. RUNNING MODEL BENCHMARK")
logging.info("=" * 80)

f1_macro = make_scorer(f1_score, average='macro')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary of models to test (reduced slightly for realistic execution time)
models_to_test = {
    'LightGBM': (lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, 
                                    min_child_samples=3, class_weight=custom_weights, 
                                    random_state=42, verbose=-1, n_jobs=-1), X_train),
    
    # 'XGBoost': (xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, 
    #                               random_state=42, eval_metric='mlogloss', n_jobs=-1), X_train),
    
    # 'CatBoost': (CatBoostClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, 
    #                               min_child_weight=1, random_state=42, eval_metric='mlogloss', n_jobs=-1), X_train),
    
    # 'RandomForest': (RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-1), X_train),
    
    # 'LogisticRegression': (LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42), X_train_sc),
    
    # 'MLP_NeuralNet': (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42), X_train_sc)
}

results = {}
oof_predictions = {}
test_predictions = {}

for name, (model, data) in models_to_test.items():
    logging.info(f"Initiating training for: {name}")
    t0 = time.time()
    
    oof_proba = np.zeros((len(y_train), 6))
    test_proba = np.zeros((len(X_test), 6))
    
    try:
        # Cross-validation loop for Out-Of-Fold predictions
        for tr_idx, val_idx in skf.split(data, y_train):
            from copy import deepcopy
            fold_model = deepcopy(model)
            fold_model.fit(data[tr_idx], y_train[tr_idx])

            if name == 'XGBoost':
                sample_weights = np.array([custom_weights[y] for y in y_train[tr_idx]])
                model.fit(data[tr_idx], y_train[tr_idx], sample_weight=sample_weights)
            else:
                model.fit(data[tr_idx], y_train[tr_idx])

            if hasattr(fold_model, 'predict_proba'):
                oof_proba[val_idx] = fold_model.predict_proba(data[val_idx])
                test_proba += fold_model.predict_proba(X_test_sc if data is X_train_sc else X_test) / 5.0
            
        elapsed = time.time() - t0
        oof_pred = np.argmax(oof_proba, axis=1)
        score = f1_score(y_train, oof_pred, average='macro')

        logging.info(f"SUCCESS: {name} | OOF F1-Score (Macro) = {score:.4f} | Total Time: {elapsed:.1f}s")

        
        target_names = [k for k, v in sorted(TARGET_MAP.items(), key=lambda item: item[1])]
        report = classification_report(y_train, oof_pred, target_names=target_names)
        logging.info(f"\n--- Detailed Classification Report for {name} ---\n{report}\n{'-'*60}")
        
        results[name] = score
        oof_predictions[name] = oof_proba
        test_predictions[name] = test_proba
        
    except Exception as e:
        logging.error(f"FAILED to train {name} due to an unexpected error.", exc_info=True)

# =============================================================================
# 4. ENSEMBLING (STACKING) & SUBMISSION
# =============================================================================
logging.info("\n" + "=" * 80)
logging.info("4. BUILDING THE ENSEMBLE & GENERATING SUBMISSION")
logging.info("=" * 80)

try:
    if oof_predictions:
        # Create Meta-features from OOF predictions
        model_names = list(oof_predictions.keys())
        meta_X_train = np.hstack([oof_predictions[n] for n in model_names])
        meta_X_test = np.hstack([test_predictions[n] for n in model_names])
        # Meta-learner: Logistic Regression
        meta_model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        meta_scores = cross_val_score(meta_model, meta_X_train, y_train, cv=skf, scoring=f1_macro)

        logging.info(f"Meta-Learner (Stacking) | F1-Score = {meta_scores.mean():.4f}")
        # Train final meta-model and predict
        meta_model.fit(meta_X_train, y_train)
        final_test_preds = meta_model.predict(meta_X_test)
        # The correct ID column based on Kaggle data sample provided
        test_ids = test_df['index'].values
        submission = pd.DataFrame({
            'Id': test_ids,
            'change_type': final_test_preds.astype(int)
        })

        submission.to_csv('stacked_ensemble_submission.csv', index=False)
        logging.info("✅ Success! 'stacked_ensemble_submission.csv' is ready for Kaggle.")
    else:
        logging.warning("No models successfully trained. Skipping Ensemble and Submission.")
except Exception as e:
    logging.error("Failed during the Ensembling or CSV export phase.", exc_info=True)

logging.info("Benchmark execution finished. Check 'benchmark_execution.log' for details.")