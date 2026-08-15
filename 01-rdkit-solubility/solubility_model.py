"""
Phase 1 Baseline Project: Aqueous Solubility Prediction (Delaney ESOL)
---------------------------------------------------------------------
Featurizer: RDKit Morgan Fingerprint (ECFP4, 2048 bits)
Model: RandomForestRegressor (scikit-learn)
Evaluation: R^2 and RMSE metrics on Delaney dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# RDKit imports
from rdkit import Chem
try:
    from rdkit.Chem import rdFingerprintGenerator
    USE_NEW_FP_GEN = True
except ImportError:
    from rdkit.Chem import AllChem
    USE_NEW_FP_GEN = False

DELANEY_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

def fetch_delaney_data():
    """Fetch Delaney dataset from GitHub."""
    print("Fetching Delaney (ESOL) dataset...")
    df = pd.read_csv(DELANEY_URL)
    print(f"Dataset loaded: {len(df)} compounds found.")
    return df

if USE_NEW_FP_GEN:
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smilestofingerprint(smiles):
    """Convert SMILES string into a 2048-bit Morgan Fingerprint (ECFP4)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if USE_NEW_FP_GEN:
        fp_np = fp_gen.GetCountFingerprintAsNumPy(mol)
        return (fp_np > 0).astype(np.int8)  # Convert count to bit vector
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

def main():
    # 1. Load Data
    df = fetch_delaney_data()
    
    smiles_col = 'smiles'
    target_col = 'measured log solubility in mols per litre'
    
    # 2. Featurize Molecules
    print("Featurizing SMILES into Morgan Fingerprints (ECFP4, 2048 bits)...")
    valid_indices = []
    features = []
    targets = []
    
    for idx, row in df.iterrows():
        s = row[smiles_col]
        y = row[target_col]
        fp = smilestofingerprint(s)
        if fp is not None:
            features.append(fp)
            targets.append(y)
            valid_indices.append(idx)
            
    X = np.array(features)
    y = np.array(targets)
    print(f"Featurized shape: X={X.shape}, y={y.shape}")

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 4. Train RandomForest Regressor
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate Performance
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n" + "="*45)
    print("   MODEL EVALUATION RESULTS (Phase 1 Baseline)")
    print("="*45)
    print(f"Train R^2 Score : {train_r2:.4f}")
    print(f"Train RMSE      : {train_rmse:.4f}")
    print("-"*45)
    print(f"Test R^2 Score  : {test_r2:.4f}")
    print(f"Test RMSE       : {test_rmse:.4f}")
    print("="*45)

    # 6. Plot & Save Figure
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7, color='#3B82F6', edgecolors='k', linewidth=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')
    plt.xlabel("Actual Measured log S (mol/L)")
    plt.ylabel("Predicted log S (mol/L)")
    plt.title(f"Delaney ESOL Solubility Prediction\nRandomForest (ECFP4) - Test R²={test_r2:.3f}, RMSE={test_rmse:.3f}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("solubility_actual_vs_predicted.png", dpi=300)
    print("Saved plot to 'solubility_actual_vs_predicted.png'.")

if __name__ == "__main__":
    main()
