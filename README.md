# AI & Machine Learning for Drug Discovery

**Author**: Hardik Sood ([@hardiksood21](https://github.com/hardiksood21))  
**Focus Area**: Computational Chemistry, Cheminformatics, Molecular Representation Learning & Structure-Based Drug Design  
**Target Applications**: DAAD WISE/RISE, Mitacs Globalink & Graduate Research in Computational Biology / Bioinformatics

---

## 📌 Executive Summary

This repository documents an ongoing research portfolio applying machine learning and cheminformatics to key challenges in computational drug discovery. The project follows a disciplined, rigorous progression—from fundamental 2D topological molecular descriptors to deep graph neural networks, molecular docking, and generative AI pipelines for de novo molecule optimization.

Every module in this repository contains **fully working code, reproducible benchmarks, and scientific documentation** suitable for academic peer review and research evaluation.

---

## 🔬 Research Projects Overview

| Module | Research Domain | Benchmark / Dataset | Key Methods & Architecture | Evaluation Metrics | Implementation Status |
|:---|:---|:---|:---|:---|:---:|
| **[`01-rdkit-solubility`](./01-rdkit-solubility/)** | ADME Property Prediction | Delaney (ESOL) | RDKit, ECFP4 (Morgan Fingerprints, $r=2$, 2048-bit), Random Forest Regression | **Test $R^2 = 0.714$**<br>**Test RMSE = 1.163 $\log(\text{mol/L})$** | **Completed & Validated** |
| **[`02-deep-learning-fastai`](./02-deep-learning-fastai/)** | Deep Learning & PyTorch | Computer Vision / CIFAR | PyTorch, ResNet-18 Transfer Learning, AdamW, Cosine Annealing, Data Augmentations | **Val Accuracy = 94.5%**<br>**CrossEntropy Loss** | **Completed & Validated** |
| **[`03-deepchem-solubility`](./03-deepchem-solubility/)** | Molecular Graph Networks | Delaney (ESOL) | DeepChem, `ConvMolFeaturizer`, `GraphConvModel` (Dual GraphConv + Pooling) | **Test $R^2 = 0.785$**<br>**Test RMSE = 0.982 $\log(\text{mol/L})$** | **Completed & Validated** |
| **[`04-gnn-binding-affinity`](./04-gnn-binding-affinity/)** | Structure-Based Drug Design | PDBbind Benchmark | PyTorch Geometric (PyG), Message Passing Neural Networks (GCNConv), Global Readout | **Test $R^2 = 0.741$**<br>**Pearson $r = 0.865$** | **Completed & Validated** |

---

## 📊 Benchmark Spotlight: Traditional ML vs. Graph Convolutional Neural Networks

A core highlight of this research portfolio is the side-by-side performance evaluation on the **Delaney (ESOL)** benchmark dataset using the exact same 80/20 train-test split ($N_{\text{train}} = 902$, $N_{\text{test}} = 226$, seed `42`):

| Model Architecture | Input Representation | Training $R^2$ | Test $R^2$ | Test RMSE ($\log S$, mol/L) | Relative Improvement |
|:---|:---|:---:|:---:|:---:|:---:|
| **Phase 1**: Random Forest Baseline | ECFP4 Morgan Fingerprint (2048-bit) | `0.9403` | `0.7138` | `1.1631` | Baseline |
| **Phase 3**: DeepChem `GraphConvModel` | `ConvMol` Molecular Graph | `0.8842` | **`0.7854`** | **`0.9821`** | **$+10.0\%$ $R^2$ / $-15.6\%$ RMSE** |

---

## 🧪 Completed Projects Breakdown

### [Phase 1: Small Molecule Aqueous Solubility Prediction (Delaney ESOL)](./01-rdkit-solubility/)
- **Biological Context**: Aqueous solubility ($\log S$) is a fundamental ADME parameter determining drug absorption, bioavailability, and formulation feasibility in early-stage discovery.
- **Cheminformatics Pipeline**: SMILES parsing with RDKit, ECFP4 2048-bit Morgan Fingerprint extraction, `RandomForestRegressor` ensemble modeling.
- **Key Findings**: Baseline test performance achieved $R^2 = 0.714$ and $\text{RMSE} = 1.163$.

### [Phase 2: Practical Deep Learning & PyTorch Infrastructure](./02-deep-learning-fastai/)
- **Theoretical Foundations**: Deep neural network mechanics, PyTorch autograd engine, transfer learning using `ResNet-18`, weight decay ($L_2$) regularization, `AdamW` optimizer, and Cosine Annealing learning rate policy.
- **Deliverables**: Modular PyTorch script ([`pytorch_classifier.py`](./02-deep-learning-fastai/pytorch_classifier.py)) and validated notebook ([`pytorch_classifier.ipynb`](./02-deep-learning-fastai/pytorch_classifier.ipynb)).

### [Phase 3: DeepChem Molecular Graph Networks](./03-deepchem-solubility/)
- **Graph Representation Learning**: Converts SMILES strings directly into molecular graph structures ($G = (V, E)$) via `ConvMolFeaturizer`, extracting 75-dimensional atom feature vectors.
- **GraphConv Architecture**: Dual Graph Convolution layers (128 channels), max graph pooling, dense projection (128 units), and dropout ($p=0.2$) regularization.
- **Key Findings**: Outperformed traditional ECFP4 fingerprints, increasing test $R^2$ to **0.7854** and decreasing test RMSE to **0.9821 $\log(\text{mol/L})$**.

### [Phase 4: PyG Graph Neural Networks for Binding Affinity (PDBbind)](./04-gnn-binding-affinity/)
- **Message Passing Neural Networks (MPNN)**: Implements custom 3-layer `GCNConv` with Batch Normalization and Global Mean Pooling in PyTorch Geometric (PyG) for protein-ligand binding affinity prediction ($pK_d$).
- **Key Findings**: Achieved a test set $R^2 = \mathbf{0.7412}$, Pearson correlation $r = \mathbf{0.8653}$, and $\text{RMSE} = \mathbf{0.9418} \ pK_d \text{ units}$.

---

## 🗺️ Research Roadmap & Future Development

1. **Phase 1 (Completed)**: RDKit Cheminformatics & Topological Fingerprints (ESOL Solubility Prediction).
2. **Phase 2 (Completed)**: Deep Learning & PyTorch Infrastructure (Transfer Learning & Optimization).
3. **Phase 3 (Completed)**: DeepChem Graph Convolutional Neural Networks (`GraphConvModel`) on ESOL dataset.
4. **Phase 4 (Completed)**: Structure-Based Binding Affinity Prediction on PDBbind with PyTorch Geometric (PyG).
5. **Phase 5 (Active Target)**: Protein Structure Representation (ESM-2) & Molecular Docking Comparison (AutoDock Vina vs. DiffDock).
6. **Phase 6 (Upcoming Capstone)**: De Novo Molecule Generation & Reinforcement Learning (REINVENT Pipeline).

---

## ⚙️ Technical Stack & Dependencies

- **Cheminformatics & Graph Learning**: `PyTorch Geometric (PyG)`, `DeepChem`, `RDKit`, `TeachOpenCADD`
- **Deep Learning Frameworks**: `PyTorch`, `torchvision`, `fastai`
- **Machine Learning & Analytics**: `scikit-learn`, `pandas`, `numpy`, `scipy`
- **Visualization**: `matplotlib` (publication quality)
- **Environment**: Python 3.10+, Google Colab, Jupyter Notebooks

---

## 📬 Contact & Academic Inquiries

If you are a Principal Investigator (PI) reviewing this repository for research internships (DAAD WISE, Mitacs Globalink) or MS admissions in Computational Biology / Bioinformatics, please feel free to inspect the detailed project modules or contact me via GitHub.
