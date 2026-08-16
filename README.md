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

---

## 🧪 Completed Projects Breakdown

### [Phase 1: Small Molecule Aqueous Solubility Prediction (Delaney ESOL)](./01-rdkit-solubility/)
- **Biological Context**: Aqueous solubility ($\log S$) is a fundamental ADME parameter determining drug absorption, bioavailability, and formulation feasibility in early-stage discovery.
- **Cheminformatics Pipeline**:
  1. Parsing chemical structures from SMILES representations into RDKit `Mol` graph objects.
  2. Generating Extended-Connectivity Fingerprints (**ECFP4**, radius $r=2$) hashed to 2048-bit circular bit vectors.
  3. Training an ensemble `RandomForestRegressor` with 80/20 train-test splitting.
- **Key Findings**: Achieved an $R^2$ of **0.714** and RMSE of **1.163** on the independent test split (226 compounds).

### [Phase 2: Practical Deep Learning & PyTorch Infrastructure](./02-deep-learning-fastai/)
- **Theoretical Foundations**:
  1. Deep neural network mechanics: Tensors, autograd gradient tracking (`loss.backward()`), backpropagation.
  2. Computer Vision Feature Extraction: Pre-trained convolutional backbones (`ResNet-18`) adapted with custom classification heads.
  3. Regularization & Optimization: Weight Decay ($L_2$), Dropout ($p=0.3$), `AdamW` optimizer with Cosine Annealing learning rate schedule.
- **Deliverables**: Modular PyTorch script ([`pytorch_classifier.py`](./02-deep-learning-fastai/pytorch_classifier.py)) and validated Google Colab notebook ([`pytorch_classifier.ipynb`](./02-deep-learning-fastai/pytorch_classifier.ipynb)).

---

## 🗺️ Research Roadmap & Future Development

1. **Phase 1 (Completed)**: RDKit Cheminformatics & Topological Fingerprints (ESOL Solubility Prediction).
2. **Phase 2 (Completed)**: Deep Learning & PyTorch Infrastructure (Transfer Learning & Optimization).
3. **Phase 3 (Active Target)**: DeepChem Graph Convolutional Neural Networks (`GraphConvModel`) on ESOL dataset for direct baseline comparison against Phase 1.
4. **Phase 4 (Upcoming)**: Structure-Based Binding Affinity Prediction on PDBbind with PyTorch Geometric (PyG).
5. **Phase 5 (Upcoming)**: Protein Structure Representation (ESM-2) & Molecular Docking Comparison (AutoDock Vina vs. DiffDock).
6. **Phase 6 (Upcoming Capstone)**: De Novo Molecule Generation & Reinforcement Learning (REINVENT Pipeline).

---

## ⚙️ Technical Stack & Dependencies

- **Cheminformatics & Molecular Processing**: `RDKit`, `TeachOpenCADD`
- **Deep Learning Frameworks**: `PyTorch`, `torchvision`, `fastai`
- **Machine Learning & Analytics**: `scikit-learn`, `pandas`, `numpy`
- **Visualization**: `matplotlib` (publication quality)
- **Environment**: Python 3.10+, Google Colab, Jupyter Notebooks

---

## 📬 Contact & Academic Inquiries

If you are a Principal Investigator (PI) reviewing this repository for research internships (DAAD WISE, Mitacs Globalink) or MS admissions in Computational Biology / Bioinformatics, please feel free to inspect the detailed project modules or contact me via GitHub.
