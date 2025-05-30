import pandas as pd
from preprocessing import preprocess_test
import joblib



def generate_submission(test_csv_path, best_model, output_path='submission.csv'):
    df_test = pd.read_csv(test_csv_path)
    df_test_clean = preprocess_test(test_csv_path)
    feature_cols = [c for c in df_test_clean.columns if c.endswith('_enc')] + ['motor_volume', 'running_miles']
    X_test = df_test_clean[feature_cols]
    preds = best_model.predict(X_test)
    submission = pd.DataFrame({
        'id': df_test['Id'],
        'price': preds
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission salvata in: {output_path}")
    
    
generate_submission('CarPrice/data/test.csv', joblib.load('CarPrice/models/XGBoost_best_model.joblib'), 'CarPrice/data/submission.csv')




