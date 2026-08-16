"""
Phase 3 Project: DeepChem Graph Convolutional Neural Network (Delaney ESOL)
----------------------------------------------------------------------------
Featurizer: ConvMolFeaturizer (Molecular Graph Representation)
Model: GraphConvModel (DeepChem Graph Convolutional Architecture)
Benchmark Comparison: Phase 1 Random Forest Baseline vs. Phase 3 GraphConv
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# DeepChem imports
import deepchem as dc
from deepchem.feat import ConvMolFeaturizer
from deepchem.models import GraphConvModel
from deepchem.metrics import Metric, pearson_r2_score, rms_score

DELANEY_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

def load_and_featurize_delaney():
    """Load Delaney dataset and featurize SMILES into ConvMol graph representations."""
    print("Fetching Delaney (ESOL) dataset...")
    df = pd.read_csv(DELANEY_URL)
    print(f"Loaded {len(df)} compounds.")
    
    print("Featurizing SMILES into ConvMol molecular graph representations...")
    featurizer = ConvMolFeaturizer()
    features = featurizer.featurize(df['smiles'].tolist())
    targets = df['measured log solubility in mols per litre'].to_numpy()
    
    # Create DeepChem NumpyDataset
    dataset = dc.data.NumpyDataset(X=features, y=targets, ids=df['smiles'].tolist())
    
    # Split using RandomSplitter (80/20 split, seed=42) for exact alignment with Phase 1
    splitter = dc.splits.RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8, seed=42)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset

def main():
    # 1. Load & Featurize Data
    train_dataset, test_dataset = load_and_featurize_delaney()
    
    # 2. Build GraphConvModel Architecture
    print("Initializing DeepChem GraphConvModel...")
    model = GraphConvModel(
        n_tasks=1,
        mode='regression',
        dropout=0.2,
        dense_layer_size=128,
        graph_conv_layers=[128, 128],
        random_seed=42
    )
    
    # 3. Train GraphConvModel
    print("Training GraphConvModel for 60 epochs...")
    metric_r2 = Metric(pearson_r2_score, name="r2")
    metric_rmse = Metric(rms_score, name="rmse")
    
    losses = []
    epochs = 60
    for epoch in range(1, epochs + 1):
        loss = model.fit(train_dataset, nb_epoch=1)
        losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {loss:.4f}")
            
    # 4. Evaluate Performance
    print("\nEvaluating GraphConvModel performance on train and test sets...")
    train_y_true = train_dataset.y
    train_y_pred = model.predict(train_dataset).flatten()
    
    test_y_true = test_dataset.y
    test_y_pred = model.predict(test_dataset).flatten()
    
    train_r2 = r2_score(train_y_true, train_y_pred)
    train_rmse = np.sqrt(mean_squared_error(train_y_true, train_y_pred))
    
    test_r2 = r2_score(test_y_true, test_y_pred)
    test_rmse = np.sqrt(mean_squared_error(test_y_true, test_y_pred))
    
    print("\n" + "="*60)
    print("   MODEL EVALUATION RESULTS (Phase 3 DeepChem GraphConv)")
    print("="*60)
    print(f"Train R^2 Score : {train_r2:.4f}")
    print(f"Train RMSE      : {train_rmse:.4f}")
    print("-"*60)
    print(f"Test R^2 Score  : {test_r2:.4f}")
    print(f"Test RMSE       : {test_rmse:.4f}")
    print("="*60)
    
    # Print Benchmark Comparison Table
    print("\n" + "="*65)
    print("   BENCHMARK COMPARISON: Phase 1 Baseline vs Phase 3 GraphConv")
    print("="*65)
    print(f"{'Model Architecture':<30} | {'Test R^2':<10} | {'Test RMSE (log S)':<15}")
    print("-"*65)
    print(f"{'Phase 1: Random Forest (ECFP4)':<30} | {'0.7138':<10} | {'1.1631':<15}")
    print(f"{'Phase 3: DeepChem GraphConv':<30} | {test_r2:<10.4f} | {test_rmse:<15.4f}")
    print("="*65)
    
    # 5. Plot Comparison Scatter Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(test_y_true, test_y_pred, alpha=0.75, color='#10B981', edgecolors='k', linewidth=0.5, label='GraphConv Predictions')
    plt.plot([test_y_true.min(), test_y_true.max()], [test_y_true.min(), test_y_true.max()], 'r--', label='Ideal 1:1')
    plt.xlabel("Measured log S (mol/L)")
    plt.ylabel("Predicted log S (mol/L)")
    plt.title(f"DeepChem GraphConvModel (ESOL Solubility)\nTest R² = {test_r2:.3f}, RMSE = {test_rmse:.3f}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("deepchem_actual_vs_predicted.png", dpi=300)
    print("Saved evaluation scatter plot to 'deepchem_actual_vs_predicted.png'.")

if __name__ == "__main__":
    main()
