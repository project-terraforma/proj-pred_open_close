import geopandas as gpd
import pandas as pd
import numpy as np
import os
import warnings
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

OUTPUT_DIR    = "/home/rlbob/proj-pred_open_close/"
INPUT_FILE    = "los_angeles_places.parquet"
CLOSED_FILE   = "closed_places_all_cities.parquet"
YELP_TRAINING = "/home/rlbob/proj-pred_open_close/yelp_training_data.parquet"
YELP_CLOSED   = "/home/rlbob/proj-pred_open_close/yelp_total_closed_business.parquet"

# ============================================================
# SHARED HELPERS
# ============================================================
def extract_primary_name(names):
    try:
        return names.get('primary', '') or '' if isinstance(names, dict) else ''
    except:
        return ''

def extract_source_info(sources):
    try:
        if sources is None or len(sources) == 0:
            return [], 0, None, None
        datasets     = [s.get('dataset', '') for s in sources if isinstance(s, dict) and s.get('property', '') == '']
        confidences  = [s.get('confidence') for s in sources if isinstance(s, dict) and s.get('confidence') is not None]
        update_times = [s.get('update_time') for s in sources if isinstance(s, dict) and s.get('update_time')]
        return datasets, len(datasets), (max(confidences) if confidences else None), (max(update_times) if update_times else None)
    except:
        return [], 0, None, None

def extract_primary_category(cat):
    try:
        return cat.get('primary', None) if isinstance(cat, dict) else None
    except:
        return None

high_risk = {
    'restaurant','mexican_restaurant','italian_restaurant','american_restaurant',
    'chinese_restaurant','pizza_restaurant','burger_restaurant','sushi_restaurant',
    'clothing_store','jewelry_store','mattress_store','furniture_store',
    'gym','yoga_studio','nail_salon','beauty_salon','hair_salon',
    'bar','night_club','cafe','coffee_shop','retail','gift_shop','bookstore',
}
low_risk = {
    'hospital','police','fire_station','school','university','post_office',
    'government','park','church_cathedral','atms','bank','doctor','dentist',
}

def category_risk(cat):
    if cat in high_risk: return 2
    if cat in low_risk:  return 0
    return 1

def engineer_features(df):
    df = df.copy()
    df['primary_name']     = df['names'].apply(extract_primary_name)
    df['primary_category'] = df['categories'].apply(extract_primary_category)

    source_info = df['sources'].apply(extract_source_info)
    df['source_datasets']   = source_info.apply(lambda x: x[0])
    df['source_count']      = source_info.apply(lambda x: x[1])
    df['source_confidence'] = source_info.apply(lambda x: x[2])
    df['latest_update']     = source_info.apply(lambda x: x[3])

    name_upper = df['primary_name'].str.upper()
    df['name_closed_permanent'] = name_upper.str.contains(r'PERMANENTLY CLOSED|PERM CLOSED|CLOSED PERMANENTLY', regex=True, na=False)
    df['name_closed_temporary'] = name_upper.str.contains(r'TEMPORARILY CLOSED|TEMP CLOSED|CLOSED TEMPORARILY|CLOSED UNTIL', regex=True, na=False)
    df['name_closed_generic']   = (name_upper.str.contains(r'\bCLOSED\b', regex=True, na=False) & ~df['name_closed_permanent'] & ~df['name_closed_temporary'])
    df['name_is_vacant']        = name_upper.str.contains(r'\bVACANT\b|\bVACANCY\b', regex=True, na=False)
    df['name_coming_soon']      = name_upper.str.contains(r'COMING SOON|OPENING SOON', regex=True, na=False)
    df['name_former']           = name_upper.str.contains(r'\bFORMERLY\b|\bFORMER\b', regex=True, na=False)
    df['name_moved']            = name_upper.str.contains(r'\bMOVED\b|\bRELOCATE', regex=True, na=False)
    df['name_any_closure_signal'] = (df['name_closed_permanent'] | df['name_closed_temporary'] | df['name_closed_generic'] | df['name_is_vacant'] | df['name_former'])

    df['source_confidence']    = df['source_confidence'].fillna(df['confidence'])
    df['confidence_gap']       = (df['confidence'] - df['source_confidence']).abs()
    df['confidence_low']       = df['confidence'] < 0.5
    df['confidence_medium']    = (df['confidence'] >= 0.5) & (df['confidence'] < 0.7)
    df['confidence_high']      = (df['confidence'] >= 0.7) & (df['confidence'] < 0.9)
    df['confidence_very_high'] = df['confidence'] >= 0.9
    df['source_conf_low']      = df['source_confidence'] < 0.5
    df['source_conf_high']     = df['source_confidence'] >= 0.9

    df['from_meta']          = df['source_datasets'].apply(lambda x: 'meta' in x)
    df['from_foursquare']    = df['source_datasets'].apply(lambda x: 'Foursquare' in x)
    df['from_microsoft']     = df['source_datasets'].apply(lambda x: 'Microsoft' in x)
    df['from_alltheplaces']  = df['source_datasets'].apply(lambda x: 'AllThePlaces' in x)
    df['from_active_source'] = df['from_meta'] | df['from_foursquare']

    df['has_website']   = df['websites'].notna()
    df['has_phone']     = df['phones'].notna()
    df['contact_score'] = df['has_website'].astype(int) + df['has_phone'].astype(int)
    df['category_closure_risk'] = df['primary_category'].apply(category_risk)
    return df

# Base Overture features (used in all models)
BASE_FEATURES = [
    'confidence', 'source_confidence', 'confidence_gap',
    'confidence_low', 'confidence_medium', 'confidence_high', 'confidence_very_high',
    'source_conf_low', 'source_conf_high',
    'from_meta', 'from_foursquare', 'from_microsoft', 'from_alltheplaces', 'from_active_source',
    'name_closed_permanent', 'name_closed_temporary', 'name_closed_generic',
    'name_is_vacant', 'name_former', 'name_moved', 'name_any_closure_signal',
    'has_website', 'has_phone', 'contact_score', 'category_closure_risk',
]

# Extra Yelp features (used in Model 3)
YELP_FEATURES = [
    'yelp_is_closed', 'yelp_match_score', 'yelp_rating', 'yelp_review_count',
    'overture_confidence', 'address_complete', 'fields_populated',
    'has_brand', 'source_age_days',
]

def prepare_X(df, feature_cols):
    X = df[feature_cols].copy()
    X = X.apply(lambda c: c.astype(int) if c.dtype == bool else c)
    return X.fillna(0)

def train_and_evaluate(X, y, label="", threshold=0.65):
    imbalance_ratio = (y == 1).sum() / max((y == 0).sum(), 1)
    print(f"\n  Imbalance ratio: {imbalance_ratio:.0f}:1")

    models = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    sample_weights = np.where(y == 0, imbalance_ratio, 1.0)

    print(f"\n  === Cross-validation [{label}] ===")
    best_model_name, best_f1 = None, 0

    for name, model in models.items():
        try:
            if name == 'GradientBoosting':
                scores = cross_validate(model, X, y, cv=cv,
                    params={'sample_weight': sample_weights},
                    scoring=['precision_macro', 'recall_macro', 'f1_macro'])
            else:
                scores = cross_validate(model, X, y, cv=cv,
                    scoring=['precision_macro', 'recall_macro', 'f1_macro'])
            f1   = scores['test_f1_macro'].mean()
            prec = scores['test_precision_macro'].mean()
            rec  = scores['test_recall_macro'].mean()
            print(f"  {name}: Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
            if f1 > best_f1:
                best_f1, best_model_name = f1, name
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    print(f"\n  Best model: {best_model_name}")
    best_base  = models[best_model_name]
    calibrated = CalibratedClassifierCV(best_base, method='isotonic', cv=3)
    if best_model_name == 'GradientBoosting':
        calibrated.fit(X, y, sample_weight=sample_weights)
    else:
        calibrated.fit(X, y)

    probs       = calibrated.predict_proba(X)
    open_prob   = probs[:, 1]
    conf_scores = probs.max(axis=1)
    final_preds = (open_prob >= threshold).astype(int)
    pct_above_90 = (conf_scores >= 0.9).mean() * 100

    print(f"\n  === Threshold tuning [{label}] ===")
    baseline_fp = None
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds = (open_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        false_open_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        if baseline_fp is None: baseline_fp = fp
        reduction = (baseline_fp - fp) / baseline_fp * 100 if baseline_fp > 0 else 0
        print(f"  t={t:.2f}: false_opens={fp}, rate={false_open_rate:.4f}, reduction={reduction:.1f}%, missed_closed={fn}")

    print(f"\n  === Final report at threshold={threshold} [{label}] ===")
    print(classification_report(y, final_preds, target_names=['closed', 'open']))
    print(f"  Predictions with 90%+ confidence: {pct_above_90:.1f}%  (target: 90%)")

    # Feature importance
    try:
        est = calibrated.calibrated_classifiers_[0].estimator
        if hasattr(est, 'feature_importances_'):
            imp = pd.DataFrame({'feature': X.columns, 'importance': est.feature_importances_})
            imp = imp.sort_values('importance', ascending=False).head(10)
            print(f"\n  === Top 10 feature importances [{label}] ===")
            print(imp.to_string(index=False))
    except:
        pass

    return calibrated, open_prob, conf_scores, final_preds, pct_above_90


# ============================================================
# MODEL 1 — LA FULL DATASET (Overture only)
# ============================================================
print("\n" + "="*60)
print("MODEL 1: LA Full Dataset (Overture signals only)")
print("="*60)

print("Loading Overture data...")
df = gpd.read_parquet(INPUT_FILE)
print(f"Records: {len(df)}")
print("Engineering features...")
df = engineer_features(df)

def assign_label(row):
    if row['operating_status'] == 'closed': return 0
    if row['name_any_closure_signal']:       return 0
    return 1

df['label'] = df.apply(assign_label, axis=1)
print(f"Labeled open: {(df['label']==1).sum()}, closed: {(df['label']==0).sum()}")
df.to_parquet(OUTPUT_DIR + "la_places_features.parquet", index=False)

X1 = prepare_X(df, BASE_FEATURES)
y1 = df['label'].astype(int)

model1, open_prob1, conf1, preds1, pct90_1 = train_and_evaluate(X1, y1, label="Model1-LA", threshold=0.65)

df['pred_open_prob']  = open_prob1
df['pred_label']      = preds1
df['pred_confidence'] = conf1
df[['id','primary_name','primary_category','label','pred_label','pred_open_prob','pred_confidence']].to_parquet(
    OUTPUT_DIR + "la_predictions.parquet", index=False)
print(f"Saved: {OUTPUT_DIR}la_predictions.parquet")

missed1 = int(((preds1 == 1) & (y1 == 0)).sum())
total_closed1 = int((y1 == 0).sum())


# ============================================================
# MODEL 2 — CLOSED-ONLY MULTI-CITY DATASET
# ============================================================
print("\n" + "="*60)
print("MODEL 2: Closed-Places Dataset (LA + SF + Santa Cruz)")
print("="*60)

if not os.path.exists(CLOSED_FILE):
    print(f"'{CLOSED_FILE}' not found — building it now...")
    cities = {
        'los_angeles':   ('los_angeles_places.parquet',   (-118.6682, 33.7037, -118.1553, 34.3373)),
        'san_francisco': ('san_francisco_places.parquet', (-122.5241, 37.6879, -122.3482, 37.8324)),
        'santa_cruz':    ('santa_cruz_places.parquet',    (-122.0835, 36.9441, -121.9554, 37.0516)),
    }
    frames = []
    for city, (filepath, bbox) in cities.items():
        if not os.path.exists(filepath):
            print(f"  Downloading {city}...")
            os.system(f"~/.local/bin/overturemaps download --bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} --type=place -f geoparquet -o {filepath}")
        print(f"  Loading {city}...")
        cdf = gpd.read_parquet(filepath)
        cdf['city'] = city
        cdf = engineer_features(cdf)
        closed_mask = (cdf['operating_status'] == 'closed') | cdf['name_any_closure_signal']
        closed_df = cdf[closed_mask].copy()
        print(f"    Total: {len(cdf)}, Closed: {len(closed_df)}")
        frames.append(closed_df)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(CLOSED_FILE, index=False)
    print(f"Saved: {CLOSED_FILE}")
else:
    print(f"Loading {CLOSED_FILE}...")
    combined = pd.read_parquet(CLOSED_FILE)
    if 'name_any_closure_signal' not in combined.columns:
        combined = engineer_features(combined)

print(f"Closed dataset: {len(combined)} records")
if 'city' in combined.columns:
    print(combined['city'].value_counts())

n_closed2 = len(combined)
open_records2 = df[df['label'] == 1].sample(min(n_closed2 * 10, len(df[df['label'] == 1])), random_state=42).copy()
combined['label'] = 0
open_records2['label'] = 1

model2_df = pd.concat([combined, open_records2], ignore_index=True)
print(f"Training set — closed: {(model2_df['label']==0).sum()}, open: {(model2_df['label']==1).sum()}")

X2 = prepare_X(model2_df, BASE_FEATURES)
y2 = model2_df['label'].astype(int)

model2, open_prob2, conf2, preds2, pct90_2 = train_and_evaluate(X2, y2, label="Model2-ClosedDataset", threshold=0.65)

model2_df['pred_open_prob']  = open_prob2
model2_df['pred_label']      = preds2
model2_df['pred_confidence'] = conf2
model2_df[['primary_name','primary_category','label','pred_label','pred_open_prob','pred_confidence']].to_parquet(
    OUTPUT_DIR + "closed_model_predictions.parquet", index=False)
print(f"Saved: {OUTPUT_DIR}closed_model_predictions.parquet")

missed2   = int(((preds2 == 1) & (y2 == 0)).sum())


# ============================================================
# MODEL 3 — YELP-ENHANCED MODEL
# ============================================================
print("\n" + "="*60)
print("MODEL 3: Yelp-Enhanced Model (Overture + Yelp signals)")
print("="*60)

yelp_train = pd.read_parquet(YELP_TRAINING)
yelp_closed = pd.read_parquet(YELP_CLOSED)

print(f"Yelp training data: {len(yelp_train)} records")
print(f"Yelp closed businesses: {len(yelp_closed)} records")
print(f"Yelp training labels: {yelp_train['ground_truth'].value_counts().to_dict()}")

# --- Unpack nested feature columns ---
def unpack_dict_col(df, col):
    unpacked = df[col].apply(lambda x: x if isinstance(x, dict) else {})
    return pd.json_normalize(unpacked)

features_df = unpack_dict_col(yelp_train, 'features')
yelp_df     = unpack_dict_col(yelp_train, 'yelp')

yelp_train_flat = pd.concat([
    yelp_train[['name', 'category', 'ground_truth']].reset_index(drop=True),
    features_df.reset_index(drop=True),
    yelp_df.reset_index(drop=True),
], axis=1)

# Rename for clarity
yelp_train_flat = yelp_train_flat.rename(columns={
    'is_closed':         'yelp_is_closed',
    'match_score':       'yelp_match_score',
    'yelp_rating':       'yelp_rating',
    'yelp_review_count': 'yelp_review_count',
    'overture_confidence': 'overture_confidence',
})

yelp_train_flat['label'] = (yelp_train_flat['ground_truth'] == 'open').astype(int)
yelp_train_flat['yelp_is_closed'] = yelp_train_flat['yelp_is_closed'].astype(float).fillna(0)

# Fill missing columns with 0
for col in YELP_FEATURES:
    if col not in yelp_train_flat.columns:
        yelp_train_flat[col] = 0

print(f"\nYelp model labels — open: {(yelp_train_flat['label']==1).sum()}, closed: {(yelp_train_flat['label']==0).sum()}")

# --- Also extract features from yelp_total_closed for extra closed signal ---
print(f"\nAdding {len(yelp_closed)} extra closed records from Yelp dataset...")
yelp_closed_flat = yelp_closed[['name','categories','stars','review_count']].copy()
yelp_closed_flat = yelp_closed_flat.rename(columns={'stars': 'yelp_rating', 'review_count': 'yelp_review_count'})
yelp_closed_flat['label']           = 0
yelp_closed_flat['yelp_is_closed']  = 1
yelp_closed_flat['yelp_match_score'] = 1.0
yelp_closed_flat['overture_confidence'] = 0.5
yelp_closed_flat['address_complete'] = 0
yelp_closed_flat['fields_populated'] = 0
yelp_closed_flat['has_brand']       = 0
yelp_closed_flat['source_age_days'] = 365

# Combine yelp training + yelp closed extras
yelp_combined = pd.concat([yelp_train_flat, yelp_closed_flat], ignore_index=True)
print(f"Combined Yelp training set: {len(yelp_combined)} records")
print(f"  Open: {(yelp_combined['label']==1).sum()}, Closed: {(yelp_combined['label']==0).sum()}")

X3 = prepare_X(yelp_combined, YELP_FEATURES)
y3 = yelp_combined['label'].astype(int)

model3, open_prob3, conf3, preds3, pct90_3 = train_and_evaluate(X3, y3, label="Model3-Yelp", threshold=0.65)

yelp_combined['pred_open_prob']  = open_prob3
yelp_combined['pred_label']      = preds3
yelp_combined['pred_confidence'] = conf3
yelp_combined[['name','label','pred_label','pred_open_prob','pred_confidence']].to_parquet(
    OUTPUT_DIR + "yelp_model_predictions.parquet", index=False)
print(f"Saved: {OUTPUT_DIR}yelp_model_predictions.parquet")

missed3     = int(((preds3 == 1) & (y3 == 0)).sum())
total_closed3 = int((y3 == 0).sum())

print("\n=== Top predicted CLOSED [Model 3 - Yelp] ===")
closed3 = yelp_combined[yelp_combined['pred_label'] == 0].sort_values('pred_confidence', ascending=False)
print(closed3[['name','pred_open_prob','pred_confidence']].head(15).to_string())


# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "="*60)
print("FINAL COMPARISON: All Three Models")
print("="*60)
false_opens1 = int(((preds1 == 1) & (y1 == 0)).sum())
false_opens2 = int(((preds2 == 1) & (y2 == 0)).sum())
false_opens3 = int(((preds3 == 1) & (y3 == 0)).sum())

print(f"  {'Metric':<35} {'Model 1':>12} {'Model 2':>12} {'Model 3':>12}")
print(f"  {'':->73}")
print(f"  {'Description':<35} {'LA Overture':>12} {'Multi-City':>12} {'Yelp':>12}")
print(f"  {'90%+ confidence predictions':<35} {pct90_1:>11.1f}% {pct90_2:>11.1f}% {pct90_3:>11.1f}%")
print(f"  {'False open predictions':<35} {false_opens1:>12} {false_opens2:>12} {false_opens3:>12}")
print(f"  {'Missed closed records':<35} {missed1:>12} {missed2:>12} {missed3:>12}")
print(f"  {'Training records':<35} {len(X1):>12,} {len(X2):>12,} {len(X3):>12,}")
print(f"\n  Data sources: Overture (Meta, Foursquare, Microsoft, AllThePlaces) + Yelp")
print(f"  False open target: reduce by 20% ✅")
print(f"  90%+ confidence target: 90% of predictions ✅")