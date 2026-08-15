# Phase 1: Aqueous Solubility Prediction (Delaney ESOL)

**Topic**: RDKit + Cheminformatics  
**Primary Resource**: [TeachOpenCADD (Volkamer Lab)](https://github.com/volkamerlab/teachopencadd)  
**Goal**: Predict log aqueous solubility ($\log S$, mol/L) from chemical structure (SMILES) using Morgan Fingerprints (ECFP4) and Random Forest Regression.

---

## 1. Core Concept: Why Fingerprints & Bit Vectors?

Computers cannot process raw SMILES strings directly as numeric vectors for standard Machine Learning algorithms (like Random Forest or XGBoost). We must convert 2D molecular structures into fixed-length numeric vectors.

### What is a Morgan Fingerprint (ECFP4)?
- **Extended-Connectivity Fingerprints (ECFP)** are circular topological fingerprints designed for molecular characterization.
- **Radius 2 (ECFP4)**: For every atom in a molecule, the algorithm looks at its immediate neighbors (radius 1) and neighbors of neighbors (radius 2, corresponding to a diameter of 4 bonds).
- **Substructure Hashing**: Each circular atom environment is hashed into an integer identifier.
- **Bit Vector Reduction**: To create a uniform vector for machine learning models, all hash integers are mapped onto a fixed-size binary vector (**2,048 bits** using modulo arithmetic).
  - Bit = `1`: Substructure feature is **present** in the molecule.
  - Bit = `0`: Substructure feature is **absent**.

---

## 2. Dataset: Delaney (ESOL)
- **Delaney Dataset**: Standard benchmark containing 1,128 small molecules with experimentally measured log aqueous solubility ($\log S$).
- **Features**: SMILES representation of each compound.
- **Target ($\log S$)**: Logarithm of solubility measured in moles per liter.

---

## 3. Machine Learning Model & Pipeline
1. **SMILES Parsing**: Load molecules using RDKit (`Chem.MolFromSmiles`).
2. **Featurization**: Calculate 2048-bit Morgan Fingerprints ($r=2$) for each compound.
3. **Data Splitting**: 80% Training set (902 samples), 20% Test set (226 samples), `random_state=42`.
4. **Model**: `sklearn.ensemble.RandomForestRegressor(n_estimators=100)`.
5. **Metrics**:
   - $R^2$ Score (Coefficient of Determination)
   - RMSE (Root Mean Squared Error)

---

## 4. Empirical Evaluation Results

```
=============================================
   MODEL EVALUATION RESULTS (Phase 1 Baseline)
=============================================
Train R^2 Score : 0.9403
Train RMSE      : 0.5072
---------------------------------------------
Test R^2 Score  : 0.7138
Test RMSE       : 1.1631
=============================================
```

- **Scatter Plot**: Saved to [`solubility_actual_vs_predicted.png`](./solubility_actual_vs_predicted.png).

---

## 5. Files & Quick Start

- `solubility_model.py`: Standalone Python execution script.
- `solubility_model.ipynb`: Interactive Jupyter / Google Colab notebook.

To run locally:
```bash
python solubility_model.py
```
