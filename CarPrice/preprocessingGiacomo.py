import re
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Funzione per convertire 'running' in miles (rimuove unità e converte km->miles)
def convert_running_to_miles(val):
    s = str(val).strip().lower()
    # Estrai numero
    num_match = re.search(r"([0-9,.]+)", s)
    if not num_match:
        return np.nan
    num = float(num_match.group().replace(',', ''))
    # Verifica unità
    if 'km' in s:
        num = num * 0.621371  # km to miles
    # Se 'miles' o unità mancante, assume sia già miles
    return num

# Funzione per eliminare feature basate su VIF e p-value
def elimina_variabili_vif_pvalue(X, y, vif_threshold=5.0, pvalue_threshold=0.05):
    X_current = X.copy()
    while True:
        X_const = sm.add_constant(X_current)
        model = sm.OLS(y, X_const).fit()
        pvals = model.pvalues.drop('const')
        vif_data = pd.DataFrame({
            'Feature': X_current.columns,
            'VIF': [variance_inflation_factor(X_current.values, i)
                    for i in range(X_current.shape[1])],
            'p-value': pvals.values
        })
        print(vif_data)
        # Condizioni di rimozione
        cond = (vif_data['VIF'] > vif_threshold) & (vif_data['p-value'] > pvalue_threshold)
        if not cond.any():
            break
        # Rimuovi feature con VIF più alto tra quelle da eliminare
        to_remove = vif_data.loc[cond, 'Feature'].iloc[vif_data.loc[cond,'VIF'].argmax()]
        print(f"Rimuovo '{to_remove}' (VIF={vif_data.loc[vif_data.Feature==to_remove,'VIF'].values[0]:.2f}, p={vif_data.loc[vif_data.Feature==to_remove,'p-value'].values[0]:.4f})")
        X_current.drop(columns=[to_remove], inplace=True)
    print("Feature finali:", X_current.columns.tolist())
    return X_current

# Preprocessing principale
if __name__ == '__main__':
    # Carica dati
    df = pd.read_csv('CarPrice/train.csv')

    # 1. Converti 'running' in miles
    df['running_miles'] = df['running'].apply(convert_running_to_miles)

    # 2. Label Encoding per colonne categoriche
    cols_to_encode = ['model', 'motor_type', 'wheel', 'color', 'status']
    le_dict = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Mappatura '{col}':", le_dict[col])
        
    # 2.1 Salvataggio train in cleaned_train.csv
    df.to_csv('cleaned_train.csv', index=False) 
    
    # 3. Preparazione features per VIF
    feature_cols = [col + '_enc' for col in cols_to_encode] + ['motor_volume', 'running_miles', 'price']
    X = df[feature_cols]
    y = df['price']

    # 4. Eliminazione variabili collineari e non significative
    print("\n--- Analisi VIF e p-value ---")
    X_selected = elimina_variabili_vif_pvalue(X, y)

    
    
    # 5. Matrice di correlazione
    print("\n--- Matrice di Correlazione ---")
    corr_matrix = df[X_selected.columns].corr()
    print(corr_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
