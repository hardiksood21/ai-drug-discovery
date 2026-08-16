# AI & Machine Learning for Drug Discovery: A Multi-Scale Computational Benchmark

**Author**: Hardik Sood ([@hardiksood21](https://github.com/hardiksood21))  
**Domain**: Computational Chemistry, Cheminformatics, Molecular Graph Representation Learning, Structural Biology & Generative AI  

---

## 📌 Executive Summary

This repository presents a rigorous, 6-phase research portfolio spanning the entire computational drug discovery paradigm: from fundamental 2D topological molecular descriptors to deep geometric graph neural networks, macromolecular docking, and closed-loop reinforcement learning for de novo molecular design.

Every phase contains **reproducible Python scripts, interactive Google Colab notebooks (validated JSON), publication-quality vector graphics (300 DPI), and formal scientific documentation**.

```
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                           END-TO-END DISCOVERY PIPELINE                                │
  └────────────────────────────────────────────────────────────────────────────────────────┘
     Phase 1: 2D Cheminformatics (RDKit Morgan Fingerprints / ECFP4)
                           │
                           ▼
     Phase 2: Deep Learning Infrastructure (PyTorch, Autograd, Transfer Learning)
                           │
                           ▼
     Phase 3: Molecular Graph Learning (DeepChem GraphConvModel vs. Phase 1 RF)
                           │
                           ▼
     Phase 4: Structure-Based Graph Learning (PyTorch Geometric MPNN on PDBbind)
                           │
                           ▼
     Phase 5: Macromolecular Structural Biology (ESM-2 Embeddings & Vina vs. DiffDock)
                           │
                           ▼
     Phase 6: Closed-Loop De Novo Generation (REINVENT Policy Gradient + Phase 3 Reward)
```

---

## 🔬 Portfolio Modules & Benchmark Summary

| Module | Research Domain | Benchmark / Target | Primary Methods & Architecture | Key Evaluation Metrics | Status |
|:---|:---|:---|:---|:---|:---:|
| **[`01-rdkit-solubility`](./01-rdkit-solubility/)** | ADME Property Prediction | Delaney (ESOL) | RDKit, ECFP4 (Morgan Fingerprints, $r=2$, 2048-bit), Random Forest Regression | **Test $R^2 = 0.714$**<br>**Test RMSE = 1.163 $\log(\text{mol/L})$** | **Completed & Validated** |
| **[`02-deep-learning-fastai`](./02-deep-learning-fastai/)** | Deep Learning & PyTorch | Computer Vision / CIFAR | PyTorch, ResNet-18 Transfer Learning, AdamW, Cosine Annealing, Data Augmentations | **Val Accuracy = 94.5%**<br>**CrossEntropy Loss** | **Completed & Validated** |
| **[`03-deepchem-solubility`](./03-deepchem-solubility/)** | Molecular Graph Networks | Delaney (ESOL) | DeepChem, `ConvMolFeaturizer`, `GraphConvModel` (Dual GraphConv + Pooling) | **Test $R^2 = 0.785$**<br>**Test RMSE = 0.982 $\log(\text{mol/L})$** | **Completed & Validated** |
| **[`04-gnn-binding-affinity`](./04-gnn-binding-affinity/)** | Structure-Based Drug Design | PDBbind Benchmark | PyTorch Geometric (PyG), Message Passing Neural Networks (GCNConv), Global Readout | **Test $R^2 = 0.741$**<br>**Pearson $r = 0.865$** | **Completed & Validated** |
| **[`05-protein-docking`](./05-protein-docking/)** | Structural Biology & Docking | EGFR Kinase (`1M17`) | ESM-2 (Protein Language Model), AutoDock Vina, DiffDock (Diffusion Pose Prediction) | **DiffDock RMSD = 0.82 Å**<br>**Vina $\Delta G = -8.9 \text{ kcal/mol}$** | **Completed & Validated** |
| **[`06-generative-reinvent`](./06-generative-reinvent/)** | De Novo Generative Design | ChEMBL Prior Agent | REINVENT Reinforcement Learning Loop, Custom Phase 3 DeepChem Reward Model | **100% Valid SMILES**<br>**$\log S > -1.9$, QED $> 0.85$** | **Completed & Validated** |

---

## 📊 Benchmark Spotlight: Traditional ML vs. Graph Convolutional Neural Networks

A key highlight of this portfolio is the direct benchmark comparison on the **Delaney (ESOL)** dataset using the exact same 80/20 train-test split ($N_{\text{train}} = 902$, $N_{\text{test}} = 226$, seed `42`):

| Model Architecture | Input Representation | Training $R^2$ | Test $R^2$ | Test RMSE ($\log S$, mol/L) | Relative Improvement |
|:---|:---|:---:|:---:|:---:|:---:|
| **Phase 1**: Random Forest Baseline | ECFP4 Morgan Fingerprint (2048-bit) | `0.9403` | `0.7138` | `1.1631` | Baseline |
| **Phase 3**: DeepChem `GraphConvModel` | `ConvMol` Molecular Graph | `0.8842` | **`0.7854`** | **`0.9821`** | **$+10.0\%$ $R^2$ / $-15.6\%$ RMSE** |

---

## 🧪 Comprehensive Project Breakdown

### [Phase 1: Small Molecule Aqueous Solubility Prediction (Delaney ESOL)](./01-rdkit-solubility/)
- **Biological Context**: Aqueous solubility ($\log S$) governs oral drug bioavailability and pharmacokinetic efficacy.
- **Cheminformatics Pipeline**: SMILES parsing with RDKit, ECFP4 2048-bit circular bit vector featurization ($r=2$), and `RandomForestRegressor` ensemble modeling.
- **Key Findings**: Baseline test performance achieved $R^2 = 0.714$ and $\text{RMSE} = 1.163$.

### [Phase 2: Practical Deep Learning & PyTorch Infrastructure](./02-deep-learning-fastai/)
- **Theoretical Foundations**: Autograd engine mechanics, transfer learning using `ResNet-18`, weight decay ($L_2$) regularization, `AdamW` optimizer, and Cosine Annealing learning rate schedule.
- **Deliverables**: Modular PyTorch script ([`pytorch_classifier.py`](./02-deep-learning-fastai/pytorch_classifier.py)) and validated notebook ([`pytorch_classifier.ipynb`](./02-deep-learning-fastai/pytorch_classifier.ipynb)).

### [Phase 3: DeepChem Molecular Graph Networks](./03-deepchem-solubility/)
- **Graph Representation Learning**: Converts SMILES directly into graph objects ($G = (V, E)$) via `ConvMolFeaturizer`, extracting 75-dimensional atom feature vectors.
- **GraphConv Architecture**: Dual Graph Convolution layers (128 channels), max graph pooling, dense projection (128 units), and dropout ($p=0.2$) regularization.
- **Key Findings**: Outperformed traditional ECFP4 fingerprints, increasing test $R^2$ to **0.7854** and decreasing test RMSE to **0.9821 $\log(\text{mol/L})$**.

### [Phase 4: PyG Graph Neural Networks for Binding Affinity (PDBbind)](./04-gnn-binding-affinity/)
- **Message Passing Neural Networks (MPNN)**: Implements custom 3-layer `GCNConv` with Batch Normalization and Global Mean Pooling in PyTorch Geometric (PyG) for protein-ligand binding affinity prediction ($pK_d$).
- **Key Findings**: Achieved a test set $R^2 = \mathbf{0.7412}$, Pearson correlation $r = \mathbf{0.8653}$, and $\text{RMSE} = \mathbf{0.9418} \ pK_d \text{ units}$.

### [Phase 5: Protein Structure (ESM-2) & Molecular Docking (Vina vs. DiffDock)](./05-protein-docking/)
- **Structural Biology Pipeline**: Extracts 1280-dimensional sequence embeddings using Meta's **ESM-2** protein language model and benchmarks classical **AutoDock Vina** against generative diffusion **DiffDock** on the EGFR Kinase Domain (`1M17`).
- **Key Findings**: DiffDock achieved sub-Angstrom pose accuracy (**$0.82\ \text{\AA}$ RMSD**) on the native Erlotinib co-crystal reference.

### [Phase 6: Generative Chemistry Capstone (REINVENT + DeepChem Reward)](./06-generative-reinvent/)
- **Closed-Loop Generative Pipeline**: Connects an RNN prior agent fine-tuned via Policy Gradient RL, guided by our custom **Phase 3 DeepChem GraphConvModel** as an external reward evaluator.
- **Multi-Parameter Optimization**: Optimizes novel SMILES for high aqueous solubility ($\log S > -1.9$), high drug-likeness (QED $> 0.85$), and favorable synthetic accessibility (SA score $< 3.5$).

---

## ⚙️ Technical Stack & Dependencies

- **Generative AI & RL**: `REINVENT`, Policy Gradient RL, Multi-Parameter Optimization (MPO)
- **Protein Structure & Docking**: `ESM-2`, `AutoDock Vina`, `DiffDock`
- **Cheminformatics & Graph Learning**: `PyTorch Geometric (PyG)`, `DeepChem`, `RDKit`, `TeachOpenCADD`
- **Deep Learning Frameworks**: `PyTorch`, `torchvision`, `fastai`
- **Machine Learning & Analytics**: `scikit-learn`, `pandas`, `numpy`, `scipy`
- **Visualization**: `matplotlib` (publication quality 300 DPI)
- **Environment**: Python 3.10+, Google Colab, Jupyter Notebooks

---

## 📬 Contact & Collaboration

Open to research discussions, scientific collaborations, and inquiries in Molecular Machine Learning, Computational Biology, and Structure-Based Drug Design. Reach out via [GitHub Issues / Discussions](https://github.com/hardiksood21/ai-drug-discovery) or email.
