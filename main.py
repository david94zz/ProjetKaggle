import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Target mapping dictionary based on instructions
TARGET_MAP = {
    'Demolition': 0, 
    'Road': 1, 
    'Residential': 2, 
    'Commercial': 3, 
    'Industrial': 4,
    'Mega Projects': 5
}

def load_data(train_path='train.geojson', test_path='test.geojson'):
    """
    Load GeoJSON files into GeoPandas DataFrames.
    """
    print("Loading datasets...")
    train_gdf = gpd.read_file(train_path)
    test_gdf = gpd.read_file(test_path)
    return train_gdf, test_gdf

def engineer_features(gdf):
    """
    Perform feature engineering on the geographical and tabular data.
    This section is crucial for Section 1 of the Edunao report.
    """
    # Work on a copy to avoid mutating the original dataframe
    df = gdf.copy()
    
    # ---------------------------------------------------------
    # 1. GEOMETRIC FEATURES
    # Project to a metric CRS (Pseudo-Mercator EPSG:3857) to calculate accurate meters
    # ---------------------------------------------------------
    df_projected = df.to_crs(epsg=3857)
    df['poly_area'] = df_projected.geometry.area
    df['poly_perimeter'] = df_projected.geometry.length
    # Compactness / Complexity metric: Perimeter / sqrt(Area)
    # Differentiates long winding roads from square residential buildings
    df['poly_complexity'] = df['poly_perimeter'] / (np.sqrt(df['poly_area']) + 1e-5)
    
    # ---------------------------------------------------------
    # 2. MULTI-VALUED CATEGORICAL FEATURES (Urban & Geo types)
    # Split comma-separated strings and apply dummy encoding (One-Hot)
    # ---------------------------------------------------------
    urban_dummies = df['urban_type'].str.get_dummies(sep=',')
    urban_dummies = urban_dummies.add_prefix('urban_')
    
    geo_dummies = df['geography_type'].str.get_dummies(sep=',')
    geo_dummies = geo_dummies.add_prefix('geo_')
    
    df = pd.concat([df, urban_dummies, geo_dummies], axis=1)
    
    # ---------------------------------------------------------
    # 3. TEMPORAL & SPECTRAL VARIANCE FEATURES (Images RGB)
    # Calculate how much colors change across the 5 dates. 
    # High variance usually implies a major change (like demolition or construction)
    # ---------------------------------------------------------
    red_cols = [f'img_red_mean_date{i}' for i in range(1, 6)]
    green_cols = [f'img_green_mean_date{i}' for i in range(1, 6)]
    blue_cols = [f'img_blue_mean_date{i}' for i in range(1, 6)]
    
    df['red_variance_over_time'] = df[red_cols].std(axis=1)
    df['green_variance_over_time'] = df[green_cols].std(axis=1)
    df['blue_variance_over_time'] = df[blue_cols].std(axis=1)
    
    # Overall brightness feature for date 1 and date 5 to calculate total change
    df['brightness_date1'] = (df['img_red_mean_date1'] + df['img_green_mean_date1'] + df['img_blue_mean_date1']) / 3
    df['brightness_date5'] = (df['img_red_mean_date5'] + df['img_green_mean_date5'] + df['img_blue_mean_date5']) / 3
    df['brightness_change_1_to_5'] = df['brightness_date5'] - df['brightness_date1']
    
    # ---------------------------------------------------------
    # 4. CONSTRUCTION STATUS FEATURES
    # Count how many unique statuses a polygon had across the dates
    # ---------------------------------------------------------
    status_cols = [f'change_status_date{i}' for i in range(5)]
    df['num_unique_statuses'] = df[status_cols].nunique(axis=1)
    
    # Drop original geometry and non-numerical/raw categorical columns
    # that have already been engineered or won't be fed directly to the model
    cols_to_drop = ['geometry', 'urban_type', 'geography_type'] 
    
    # Also drop date strings as we rely on status changes and variance instead
    date_cols = [f'date{i}' for i in range(5)]
    cols_to_drop.extend(date_cols)
    
    df = df.drop(columns=cols_to_drop)
    
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        if df[col].dtype == 'object':
         
            df[col] = df[col].fillna("Missing").astype(str)
        else:
        
            df[col] = df[col].fillna(0)  
    
    return df

def build_pipeline(X_train):
    """
    Build a scikit-learn pipeline including preprocessing and the classifier.
    """
    # Identify numerical and categorical columns dynamically
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Remove Target and ID from features if they sneaked in
    if 'change_type' in num_features: num_features.remove('change_type')
    if 'index' in num_features: num_features.remove('index')
    
    # Preprocessing: Standardize numbers, One-Hot Encode strings
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Initialize Random Forest (good default for tabular & geometric data)
    rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    return pipeline

def main():
    # 1. Load Data
    train_gdf, test_gdf = load_data('train.geojson', 'test.geojson')
    
    # 2. Extract Labels and align Target Variable
    y_train = train_gdf['change_type'].map(TARGET_MAP)
    train_gdf = train_gdf.drop(columns=['change_type'])
    
    # Save test IDs for final submission
    test_ids = test_gdf['index']
    
    # 3. Apply Feature Engineering
    print("Applying feature engineering...")
    X_train_eng = engineer_features(train_gdf)
    X_test_eng = engineer_features(test_gdf)
    
    # Ensure train and test have exactly the same dummy columns
    X_train_eng, X_test_eng = X_train_eng.align(X_test_eng, join='left', axis=1, fill_value=0)
    
    # Drop index column from training features
    X_train = X_train_eng.drop(columns=['index'])
    X_test = X_test_eng.drop(columns=['index'])
    
    # 4. Build Pipeline
    print("Building model pipeline...")
    pipeline = build_pipeline(X_train)
    
    # 5. Evaluate locally using Cross-Validation (For Report Section 2)
    print("Evaluating model using 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We use 'f1_macro' as requested by the competition metric
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    print(f"\n--- Cross-Validation Results ---")
    print(f"Mean F1-Score (Macro): {cv_scores.mean():.4f}")
    print(f"Standard Deviation:    {cv_scores.std():.4f}\n")
    
    # 6. Train on full dataset and Predict
    print("Training final model on entire training dataset...")
    pipeline.fit(X_train, y_train)
    
    print("Generating predictions for test data...")
    predictions = pipeline.predict(X_test)
    
    # 7. Create Submission File
    submission = pd.DataFrame({
        'Id': test_ids,
        'change_type': predictions
    })
    
    submission.to_csv('final_submission.csv', index=False)
    print("Success! File 'final_submission.csv' has been generated for Kaggle submission.")

if __name__ == "__main__":
    main()