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
from copy import deepcopy
import random
import os

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

RANDOM_STATE = 42

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

TARGET_MAP = {
    'Demolition': 0, 'Road': 1, 'Residential': 2,
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}

REVERSE_MAP = {v: k for k, v in TARGET_MAP.items()}

# =============================================================================
# 2. MASSIVE FEATURE ENGINEERING
# =============================================================================
logging.info("\n" + "=" * 80)
logging.info("2. FEATURE ENGINEERING")
logging.info("Motivation: Exploit ALL available information in the dataset.")
logging.info("=" * 80)

def process_chronological_features(df):
    """
    Extracts the true temporal dynamics by dynamically sorting
    the dates (and associated colors) for each row.
    """
    
    for i in range(5):
        if f'date{i}' in df.columns:
            df[f'date{i}'] = pd.to_datetime(df[f'date{i}'], format="%d-%m-%Y", errors="coerce")

    
    def extract_row_dynamics(row):
        
        dates_with_idx = []
        for i in range(5):
            date_val = row.get(f'date{i}')
            if pd.notnull(date_val):
                dates_with_idx.append((date_val, i))
        
        
        if len(dates_with_idx) < 2:
            return pd.Series(dtype=float)

        
        dates_with_idx.sort(key=lambda x: x[0])
        
        
        
        result = {}
        
        
        first_date, first_idx = dates_with_idx[0]
        last_date, last_idx = dates_with_idx[-1]
        result['total_duration'] = (last_date - first_date).days
        
        
        for color in ['red', 'green', 'blue']:
            c_first = row.get(f'img_{color}_mean_date{int(first_idx)}')
            c_last = row.get(f'img_{color}_mean_date{int(last_idx)}')
            if pd.notnull(c_first) and pd.notnull(c_last):
                result[f'{color}_mean_diff_total'] = c_last - c_first
        
        
        for step in range(len(dates_with_idx) - 1):
            d_now, idx_now = dates_with_idx[step]
            d_next, idx_next = dates_with_idx[step + 1]
            
            
            if pd.notnull(d_next) and pd.notnull(d_now):
                result[f'delta_days_step{step}'] = (d_next - d_now).days
            else:
                result[f'delta_days_step{step}'] = np.nan
            
            
            for color in ['red', 'green', 'blue']:
                c_now = row.get(f'img_{color}_mean_date{int(idx_now)}')
                c_next = row.get(f'img_{color}_mean_date{int(idx_next)}')
                if pd.notnull(c_now) and pd.notnull(c_next):
                    result[f'd_{color}_step{step}'] = c_next - c_now
                    result[f'absd_{color}_step{step}'] = abs(c_next - c_now)
                    
        return pd.Series(result)

    
    logging.info("Chronological dynamics extraction (this may takes 2-3 min :( ))...")
    chrono_features = df.apply(extract_row_dynamics, axis=1)
    
    

    
    return chrono_features

def extract_advanced_features(df, fit_encoders=None):
    """
    Extracts an exhaustive list of features from images, geometry, and temporal data.
    Exhaustive feature engineering.
    Returns (features_df, encoders_dict) if fit_encoders is None (train mode)
    Returns features_df if fit_encoders is provided (test mode)
    """
    feat = pd.DataFrame(index=df.index)
    encoders = {} if fit_encoders is None else fit_encoders
    is_train = fit_encoders is None


    # # --- GROUP A: RAW IMAGE FEATURES ---
    # # Motivation: RGB mean and std are direct visual appearance indicators

    # for i in range(1, 6):
    #     for color in ['red', 'green', 'blue']:
    #         for stat in ['mean', 'std']:
    #             col = f'img_{color}_{stat}_date{i}'
    #             if col in df.columns:
    #                 feat[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # --- GROUP B: DERIVED COLOR FEATURES ---
    # Motivation: Color indices (ratios, saturation, NDVI-like) capture physical properties
    for i in range(1, 6):
        rm, gm, bm = f'img_red_mean_date{i}', f'img_green_mean_date{i}', f'img_blue_mean_date{i}'
        rs, gs, bs = f'img_red_std_date{i}', f'img_green_std_date{i}', f'img_blue_std_date{i}'

        if rm in feat.columns and gm in feat.columns and bm in feat.columns:
            r, g, b = feat[rm], feat[gm], feat[bm]
            total = r + g + b + 1e-8

            # Brightness and Ratios
            feat[f'brightness_d{i}'] = total / 3.0
            feat[f'r_ratio_d{i}'] = r / total
            feat[f'g_ratio_d{i}'] = g / total
            feat[f'b_ratio_d{i}'] = b / total
            
            # Pseudo Vegetation and Water Indices (NDVI / NDWI approximations)
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            feat[f'saturation_d{i}'] = (mx - mn) / (mx + 1e-8)
            feat[f'ndvi_d{i}'] = (g - r) / (g + r + 1e-8) # NDVI proxy
            feat[f'ndwi_d{i}'] = (g - b) / (g + b + 1e-8) # NDWI proxy


        # Standard Deviation features (Texture proxies)
        if rs in feat.columns:
            feat[f'mean_std_d{i}'] = (feat[rs] + feat[gs] + feat[bs]) / 3.0
    
    
    # # --- GROUP C: GLOBAL STATISTICS OVER 5 DATES ---
    # for color in ['red', 'green', 'blue']:
    #     cols = [f'img_{color}_mean_date{i}' for i in range(1, 6) if f'img_{color}_mean_date{i}' in feat.columns]
    #     if cols:
    #         feat[f'{color}_mean_all'] = feat[cols].mean(axis=1)
    #         feat[f'{color}_std_all'] = feat[cols].std(axis=1)
    #         feat[f'{color}_max_all'] = feat[cols].max(axis=1)
    #         feat[f'{color}_min_all'] = feat[cols].min(axis=1)

    # --- GROUP D: POLYGON GEOMETRY ---
    # Motivation: Shape and size are key (roads are long/thin, mega projects are huge)
    if 'geometry' in df.columns and df.geometry is not None:
        geom = df.geometry.to_crs(epsg=3857) # Metric projection
        feat['area'] = geom.area
        feat['perimeter'] = geom.length
        feat['compactness'] = 4 * np.pi * feat['area'] / (feat['perimeter']**2 + 1e-8)
        feat['bbox_area'] = geom.envelope.area
        feat['extent_ratio'] = feat['area'] / (feat['bbox_area'] + 1e-8)
        # feat['centroid_x'] = geom.centroid.x
        # feat['centroid_y'] = geom.centroid.y
        
        bounds = geom.bounds
        feat['bbox_w'] = bounds['maxx'].values - bounds['minx'].values
        feat['bbox_h'] = bounds['maxy'].values - bounds['miny'].values
        feat['aspect_ratio'] = feat['bbox_w'] / (feat['bbox_h'] + 1e-8)


    # --- GROUP E: CHANGE STATUS ---
    status_cols = [f'change_status_date{i}' for i in range(1, 6)]
    existing_status = [c for c in status_cols if c in df.columns]

    if existing_status:

        df[existing_status] = df[existing_status].fillna('unknown').astype(str)
        sv =  df[existing_status].values
        feat['n_status_changes'] = np.sum(sv[:, 1:] != sv[:, :-1], axis=1)
        feat['n_unique_status'] = pd.DataFrame(sv).nunique(axis=1).values

        if is_train:
            all_statuses = np.unique(sv)
            status_map = {val: i for i, val in enumerate(all_statuses)}
            encoders['status_map'] = status_map
        else:
            status_map = encoders.get('status_map', {})

        for sc in existing_status:
            feat[f'{sc}_enc'] = df[sc].astype(str).map(status_map).fillna(-1)


    

    # --- GROUP F & G: URBAN AND GEOGRAPHY TYPES ---

    if 'urban_type' in df.columns:
        urban_split = df['urban_type'].fillna('').astype(str).apply(
            lambda x: [s.strip().lower() for s in x.split(',') if s.strip()]
        )
        if is_train:
            mlb_u = MultiLabelBinarizer()
            mlb_u.fit(urban_split)
            encoders['mlb_urban'] = mlb_u
        else:
            mlb_u = encoders['mlb_urban']
        u_df = pd.DataFrame(mlb_u.transform(urban_split), columns=[f'urb_{c}' for c in mlb_u.classes_], index=feat.index)
        feat = pd.concat([feat, u_df], axis=1)
    
    if 'geography_type' in df.columns:
        geo_split = df['geography_type'].fillna('').astype(str).apply(
            lambda x: [s.strip().lower() for s in x.split(',') if s.strip()]
        )
        if is_train:
            mlb_g = MultiLabelBinarizer()
            mlb_g.fit(geo_split)
            encoders['mlb_geography'] = mlb_g
        else:
            mlb_g = encoders['mlb_geography']
        g_df = pd.DataFrame(mlb_g.transform(geo_split), columns=[f'geo_{c}' for c in mlb_g.classes_], index=feat.index)
        feat = pd.concat([feat, g_df], axis=1)

    # --- GROUP H: TEMPORAL FEATURES (CHANGES BETWEEN DATES)---
    # Motivation: Change over time is crucial to distinguish demolition vs road vs residential
    chrono_feat = process_chronological_features(df)
    feat = pd.concat([feat, chrono_feat], axis=1)

    # --- GROUP I: CROSS INTERACTIONS ---
    if 'area' in feat.columns and 'n_status_changes' in feat.columns:
        feat['area_x_changes'] = feat['area'] * feat['n_status_changes']

    
    if is_train:
        return feat, encoders
    
    return feat

try:
    logging.info("Extracting features for Train...")
    train_feat, fitted_encoders = extract_advanced_features(train_df)

    logging.info(f"Created {train_feat.shape[1]} features for Train.")
    
    logging.info("Extracting features for Test...")
    test_feat = extract_advanced_features(test_df, fit_encoders=fitted_encoders)

    # Align columns
    common_cols = sorted(set(train_feat.columns) & set(test_feat.columns))
    train_feat = train_feat[common_cols]
    test_feat = test_feat[common_cols]

    logging.info("Cleaning inf/nan values using training means...")
    train_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    train_mean = train_feat.mean(numeric_only=True)

    train_feat.fillna(train_mean, inplace=True)
    test_feat.fillna(train_mean, inplace=True)

    # Prepare final datasets
    # For models that can handle raw features, we keep them as is. For others, we will create scaled and PCA versions.
    cat_cols = [c for c in train_feat.columns if '_enc' in c]

    for col in cat_cols:
        train_feat[col] = train_feat[col].astype(int).astype('category')
        test_feat[col] = test_feat[col].astype(int).astype('category')

    X_train_raw = train_feat.copy()
    X_test_raw = test_feat.copy()
    y_train = train_df['change_type'].map(TARGET_MAP).values

    numeric_cols = [c for c in X_train_raw.columns if c not in cat_cols]
    scaler = StandardScaler()
    X_train_sc = X_train_raw.copy()
    X_test_sc = X_test_raw.copy()
    X_train_sc[numeric_cols] = scaler.fit_transform(X_train_raw[numeric_cols])
    X_test_sc[numeric_cols] = scaler.transform(X_test_raw[numeric_cols])

    logging.info(f"Final X_train shape: {X_train_raw.shape} ({len(cat_cols)}  categorical, {len(numeric_cols)} numeric)")

    #PCA for dimensionality reduction
    logging.info("Running PCA Dimensionality Reduction...")
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    logging.info(f"PCA retained {X_train_pca.shape[1]} components for 95% variance.")
    
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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Dictionary of models to test with their corresponding training data (raw, scaled, or PCA)
models_to_test = {
    # --- LIGHTGBM ---
    'LightGBM': (lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03,num_leaves=63,device='gpu',gpu_use_dp=False,
                                     min_child_samples=3, class_weight=custom_weights, 
                                    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1), X_train_raw),

    # --- XGBOOST ---
    'XGBoost': (xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8,device='cuda',tree_method='hist',
                                  enable_categorical=True,random_state=RANDOM_STATE, n_jobs=-1), X_train_raw),
    
    # --- CATBOOST ---
    'CatBoost': (CatBoostClassifier(n_estimators=1000, learning_rate=0.05, depth=8, task_type='GPU', class_weights=list(custom_weights.values()),
                                  verbose=False,cat_features=cat_cols, random_state=RANDOM_STATE), X_train_raw),
    
    

    # --- RANDOM FOREST ---
    'RandomForest': (RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=-1), X_train_raw)

    
    # # --- K-NEAREST NEIGHBORS ---
    # 'KNN_Weighted': (KNeighborsClassifier(n_neighbors=11, weights='distance', n_jobs=-1), X_train_pca)
    
    
    
    
    
    # # --- LOGISTIC REGRESSION ---
    # 'LogisticRegression': (LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE), X_train_sc),
    
    # # --- NEURAL NETWORK (MLP) ---
    # 'MLP_NeuralNet': (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE), X_train_sc),

    

    # # --- PCA VARIANT ---
    # 'LGBM_PCA': (lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1, class_weight='balanced'), X_train_pca)
}

results = {}
oof_predictions = {}
test_predictions = {}

for name, (model, data) in models_to_test.items():
    logging.info(f"Initiating training for: {name}")
    t0 = time.time()
    
    oof_proba = np.zeros((len(y_train), 6))
    # Determine which test set to use based on the training data assigned in dictionary
    if data is X_train_sc:
        X_te = X_test_sc
    elif data is X_train_pca:
        X_te = X_test_pca
    else:
        X_te = X_test_raw

    test_proba = np.zeros((len(X_te), 6))
    
    try:
        # Cross-validation loop for Out-Of-Fold predictions
        for tr_idx, val_idx in skf.split(data, y_train):
            from copy import deepcopy
            fold_model = deepcopy(model)
            

            X_tr, y_tr = data.iloc[tr_idx].copy(), y_train[tr_idx]
            X_va = data.iloc[val_idx].copy()
            X_test_curr = X_te.copy()

            if name == 'RandomForest':
                for col in cat_cols:
                    X_tr[col] = X_tr[col].astype(float)
                    X_va[col] = X_va[col].astype(float)
                    X_test_curr[col] = X_test_curr[col].astype(float)

            fold_model.fit(X_tr, y_tr)


            if hasattr(fold_model, 'predict_proba'):
                oof_proba[val_idx] = fold_model.predict_proba(X_va)
                test_proba += fold_model.predict_proba(X_test_curr) / 5.0
            else:
                # Fallback for models without predict_proba (like some SVC configs)
                preds = fold_model.predict(X_va)
                for idx, p in zip(val_idx, preds):
                    oof_proba[idx, p] = 1.0
                
                t_preds = fold_model.predict(X_test_curr)
                for idx, p in enumerate(t_preds):
                    test_proba[idx, p] += 1.0 / 5.0
            
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
        logging.info(f"Building ensemble using: {model_names}")

        # 1. Simple Average Ensemble
        avg_oof = sum(oof_predictions[n] for n in model_names) / len(model_names)
        avg_test = sum(test_predictions[n] for n in model_names) / len(model_names)
        f1_avg = f1_score(y_train, np.argmax(avg_oof, axis=1), average='macro')
        logging.info(f"Simple Average Ensemble | OOF F1-Score = {f1_avg:.4f}")

        # 2. Meta-Learner (Stacking via Logistic Regression)
        meta_X_train = np.hstack([oof_predictions[n] for n in model_names])
        meta_X_test = np.hstack([test_predictions[n] for n in model_names])

        meta_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced')
        meta_scores = cross_val_score(meta_model, meta_X_train, y_train, cv=skf, scoring=f1_macro)

        logging.info(f"Meta-Learner (Stacking) | F1-Score = {meta_scores.mean():.4f}")
        # Train final meta-model and predict
        meta_model.fit(meta_X_train, y_train)
        final_test_preds = meta_model.predict(meta_X_test)
        # Setup Submission Export
        id_col = 'Id' if 'Id' in test_df.columns else ('id' if 'id' in test_df.columns else 'index')
        test_ids = test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))

        submission = pd.DataFrame({
            'Id': test_ids,
            'change_type': final_test_preds.astype(int)
        })

        submission.to_csv('stacked_ensemble_submission.csv', index=False)
        logging.info("SUCCESS! 'stacked_ensemble_submission.csv' is ready for Kaggle.")

        # Generate Classification Report for the Meta-Learner on Train data
        meta_train_preds = cross_val_predict(meta_model, meta_X_train, y_train, cv=skf)
        target_names = [k for k, v in sorted(TARGET_MAP.items(), key=lambda item: item[1])]
        report = classification_report(y_train, meta_train_preds, target_names=target_names)
        logging.info(f"\n--- Detailed Classification Report (Stacking) ---\n{report}\n{'-'*60}")

    else:
        logging.warning("No models successfully trained. Skipping Ensemble and Submission.")
except Exception as e:
    logging.error("Failed during the Ensembling or CSV export phase.", exc_info=True)

logging.info("Benchmark execution finished. Check 'benchmark_execution.log' for details.")