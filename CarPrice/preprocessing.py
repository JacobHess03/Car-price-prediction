import re
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------ Funzioni di Preprocessing ------------

def convert_running_to_miles(val):
    s = str(val).strip().lower()
    num_match = re.search(r"([0-9,.]+)", s)
    if not num_match:
        return np.nan
    num = float(num_match.group().replace(',', ''))
    if 'km' in s:
        num *= 0.621371
    return num


def remove_outliers_iqr(df, cols, k=1.5, show=True):
    cleaned = df.copy()
    for col in cols:
        if col not in cleaned.columns:
            continue
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - k * IQR, Q3 + k * IQR
        if (cleaned[col] < lower).any() or (cleaned[col] > upper).any():
            if show:
                plt.figure(figsize=(6,4))
                plt.boxplot(cleaned[col].dropna(), vert=False)
                plt.title(f"Boxplot di {col} (con outlier)")
                plt.xlabel(col)
                plt.show()
            cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
            if show:
                plt.figure(figsize=(6,4))
                plt.boxplot(cleaned[col].dropna(), vert=False)
                plt.title(f"Boxplot di {col} (senza outlier)")
                plt.xlabel(col)
                plt.show()
    return cleaned


def elimina_variabili_vif_pvalue(X, y, vif_threshold=5.0, pvalue_threshold=0.05):
    X_current = X.copy()
    while True:
        model = sm.OLS(y, sm.add_constant(X_current)).fit()
        pvals = model.pvalues.drop('const', errors='ignore')
        vif_list = []
        for i, feat in enumerate(X_current.columns):
            vif = variance_inflation_factor(X_current.values, i)
            pval = pvals.get(feat, np.nan)
            vif_list.append({'Feature': feat, 'VIF': vif, 'p-value': pval})
        vif_data = pd.DataFrame(vif_list)
        print(vif_data)
        cond = (vif_data['VIF'] > vif_threshold) & (vif_data['p-value'] > pvalue_threshold)
        if not cond.any():
            break
        drop_feat = vif_data.loc[cond, 'Feature'].iloc[vif_data.loc[cond, 'VIF'].argmax()]
        print(f"Rimuovo {drop_feat}")
        X_current.drop(columns=[drop_feat], inplace=True)
    print("Feature finali:", X_current.columns.tolist())
    return X_current

# ------------ Preprocessing pipeline ------------

def preprocess_train(path_csv, drop_wheel=True, show_outliers=True):
    df = pd.read_csv(path_csv)
    df['running_miles'] = df['running'].apply(convert_running_to_miles)
    cols_cat = ['model','motor_type','wheel','color','type','status']
    for col in cols_cat:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col])
    drop_cols = cols_cat + ['running']
    if drop_wheel:
        drop_cols.remove('wheel')
    df_clean = df.drop(columns=drop_cols)
    num_cols=['motor_volume','running_miles','price']
    df_clean = remove_outliers_iqr(df_clean, num_cols, show=show_outliers)
    df_clean.to_csv('CarPrice/data/cleaned_train.csv', index=False)
    feature_cols = [c+'_enc' for c in cols_cat] + ['motor_volume','running_miles']
    X = df_clean[feature_cols]
    y = df_clean['price']
    X = elimina_variabili_vif_pvalue(X, y)
    return X, y

def preprocess_test(path_csv, drop_wheel=True):
    df = pd.read_csv(path_csv)
    df['running_miles'] = df['running'].apply(convert_running_to_miles)
    cols_cat = ['model','motor_type','wheel','color','type','status']
    for col in cols_cat:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col])
    drop_cols = cols_cat + ['running']
    if drop_wheel:
        drop_cols.remove('wheel')
    df_clean = df.drop(columns=drop_cols)
   
   
    feature_cols = [c+'_enc' for c in cols_cat] + ['motor_volume','running_miles']
    X_test = df_clean[feature_cols]
    return X_test

# ------------ Modelli e Nested CV ------------

def nested_cv_evaluation(X, y):
    models = {
        'RandomForest': {
            'estimator': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 5, 10]
            }
        },
        'XGBoost': {
            'estimator': XGBRegressor(objective='reg:squarederror', random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, mp in models.items():
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
        gsearch = GridSearchCV(mp['estimator'], mp['params'], cv=inner_cv,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
        scores = cross_val_score(gsearch, X, y, cv=outer_cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
        rmse_scores = -scores
        results[name] = rmse_scores
        print(f"{name}: RMSE nested CV = {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    return results

