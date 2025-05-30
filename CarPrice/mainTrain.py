from preprocessing import preprocess_train
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
import joblib
from xgboost import XGBRegressor


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
    best_models = {}
    for name, mp in models.items():
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
        gsearch = GridSearchCV(mp['estimator'], mp['params'], cv=inner_cv,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
        scores = cross_val_score(gsearch, X, y, cv=outer_cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
        rmse_scores = -scores
        results[name] = rmse_scores
        print(f"{name}: RMSE nested CV = {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
        # Fit best model on full data for later use and save
        gsearch.fit(X, y)
        best_models[name] = gsearch.best_estimator_
        model_file = f"CarPrice/models/{name}_best_model.joblib"
        joblib.dump(gsearch.best_estimator_, model_file)
        print(f"Modello {name} salvato in: {model_file}")
    return results, best_models

# ------------ Esecuzione Principale ------------

if __name__ == '__main__':
    X, y = preprocess_train('CarPrice/data/train.csv')
    nested_results = nested_cv_evaluation(X, y)