# EDSS Classification in Multiple Sclerosis using Hybrid CNN with Privacy-Preserving Encryption

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Overview

This project presents a **hybrid deep learning framework** for automatic classification of the **Expanded Disability Status Scale (EDSS)** in patients with **Multiple Sclerosis (MS)**. The model integrates:

- **Brain MRI slices** (FLAIR, T1, T2 sequences)
- **Clinical tabular data** (demographics, symptoms, neurological assessments)

Additionally, the system implements a **privacy-preserving pipeline** using:
- **AES-GCM symmetric encryption** for tabular patient data
- **CKKS homomorphic encryption (TenSEAL)** for medical images

The model achieves **~90% accuracy** and provides **SHAP-based interpretability** to support clinical decision-making.

---

## 🧠 Problem Statement

Multiple Sclerosis is a chronic neurodegenerative disease that requires continuous monitoring of disability progression. The EDSS is the gold standard scale (0–10) for assessing disability, but its manual evaluation is subjective and time-consuming.

**Goal:** Automate EDSS classification into three severity levels:
- **Class 0:** EDSS ≤ 1.5 (low disability)
- **Class 1:** 1.5 < EDSS ≤ 3.0 (moderate disability)
- **Class 2:** EDSS > 3.0 (high disability)

**Secondary goal:** Ensure patient data privacy through encryption techniques compliant with healthcare regulations (e.g., GDPR).

---

## 📊 Dataset

- **Source:** [Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation](https://data.mendeley.com/datasets/8bctsm8jz7/1)
- **Patients:** 60 subjects
- **MRI sequences:** FLAIR, T1-weighted, T2-weighted (NIfTI format)
- **Clinical features:** Age, gender, pyramidal signs, coordination, visual deficits, motor system, gait, comorbidities, etc.
- **Total slices after extraction:** ~4,189 PNG images

> **Note:** The dataset contains sensitive personal information (name, address, tax code, etc.). All such fields are removed during preprocessing.

---

## 🛠️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/DomFalco/Classificazione-dell-EDSS-nella-sclerosi-multipla-con-modello-multimodale-e-cifratura-dei-dati.git
cd edss-ms-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries:
- `torch`, `torchvision` – deep learning
- `nibabel` – NIfTI image processing
- `pandas`, `numpy`, `scikit-learn` – data handling
- `shap` – model interpretability
- `tenseal` – homomorphic encryption (CKKS)
- `pycryptodome` – AES-GCM encryption
- `matplotlib`, `seaborn` – visualization

### 3. Convert NIfTI slices to PNG

Download and install **MRIcroGL** from [this link](https://www.nitrc.org/frs/download.php/13084/MRIcroGL_windows.zip/?i_agree=1&release_id=4685) to visualize NIfTI files.

Then run the slice extraction script:

```bash
python 01_extract_slices.py
```

This will:
- Read all `.nii` files from `Patient-XXX/` folders
- Extract each axial slice
- Save as PNG with naming convention: `{patient_id}_{modality}_slice{index}_EDSS{value}.png`
- Output directory: `dataset_slices/`

---

## 🧪 Pipeline Steps

The project is organized into **four sequential phases**, matching the code files you provided.

### Phase 1: Slice Extraction (`01_extract_slices.py`)

- Mounts Google Drive (if using Colab)
- Reads NIfTI volumes using `nibabel`
- Normalizes each slice (min-max) and saves as PNG
- Associates each slice with patient ID and EDSS score from CSV

### Phase 2: Dataset Balancing & Feature Selection (`02_dataset_balancing.py`)

- Handles missing/infinite values
- Visualizes distributions (age, EDSS, age of onset, gender)
- Computes correlation matrix (clinical + neurological variables)
- Applies **SHAP analysis** with Random Forest to identify top 20 features
- Selects hybrid feature set (common across classes + globally important)
- Saves final balanced dataset: `Patient_Info.csv`

**Selected features example:**  
`Age`, `Age_of_onset`, `Pyramidal`, `Coordination`, `Motor_System`, `Visual`, `Gait`, `Pregnancies`, `STDs`, etc.

### Phase 3: Model Training on Clear Data (`03_train_clear.py`)

#### Architecture

- **CNN branch:** 3 convolutional blocks (32 → 64 → 128 channels) with BatchNorm, LeakyReLU, Dropout (0.3–0.5)
- **Tabular branch:** 2 fully connected layers (6 → 64 → 128)
- **Cross-modality attention:** dynamically weights image vs. tabular features
- **Classifier:** 256 → 3 classes (EDSS low/moderate/high)

#### Training configuration

- Loss: **Focal Loss** (gamma=2.5, class weights: [0.9, 0.7, 1.5])
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-3)
- Scheduler: OneCycleLR (max_lr=1e-3)
- Data augmentation: random affine, horizontal flip, color jitter, Gaussian blur, random erasing
- Early stopping (patience=5)
- Batch size: 32
- Epochs: 40 (early stopping at ~25)

#### Results (Clear Data)

| Metric           | Class 0 | Class 1 | Class 2 | Average |
|-----------------|---------|---------|---------|---------|
| Accuracy         | -       | -       | -       | **90.6%** |
| Precision        | 0.91    | 0.93    | 0.88    | 0.91    |
| Recall           | 0.88    | 0.90    | 0.94    | 0.91    |
| F1-score         | 0.89    | 0.91    | 0.91    | 0.90    |
| Specificity      | 0.94    | 0.96    | 0.91    | 0.94    |
| AUC              | 0.98    | 0.99    | 0.98    | 0.98    |

### Phase 4: Encryption/Decryption & Retraining (`04_encrypt_decrypt.py` + `05_train_decrypted.py`)

#### Encryption steps

**A. Tabular data (AES-GCM)**
- Password-based key derivation (PBKDF2, 1M iterations)
- Salt + nonce + authentication tag
- Output: `Patient_Info_Extended_Encrypted.enc`

**B. MRI slices (CKKS homomorphic encryption via TenSEAL)**
- Context: poly_modulus_degree=4096, coeff_mod_bit_sizes=[30,20,30]
- Images resized to 64×64 (due to Colab GPU limits)
- Each slice encrypted to `.enc` tensor
- Checkpoint system for resumable encryption

#### Decryption (before training)

- Tabular data decrypted back to `Patient_Info_Decrypted.csv`
- Images decrypted to `dataset_slices_decrypted/` as PNG

#### Results (Decrypted Data)

| Metric           | Class 0 | Class 1 | Class 2 | Average |
|-----------------|---------|---------|---------|---------|
| Accuracy         | -       | -       | -       | **89.6%** |
| Precision        | 0.89    | 0.91    | 0.88    | 0.89    |
| Recall           | 0.84    | 0.94    | 0.90    | 0.89    |
| F1-score         | 0.86    | 0.93    | 0.89    | 0.89    |
| Specificity      | 0.94    | 0.93    | 0.93    | 0.93    |
| AUC              | 0.98    | 0.98    | 0.98    | 0.98    |

> Performance drop is minimal (~1%), confirming that encryption/decryption does not significantly degrade model quality.

#### Security feature

The best model (`best_model.pth`) is **automatically encrypted** after training using AES-GCM and the user's password. The plaintext model is then deleted, ensuring that the trained weights are never stored unencrypted.

---

## 📁 Repository Structure

```
edss-ms-classification/
│
├── 01_extract_slices.py          # NIfTI → PNG conversion
├── 02_dataset_balancing.py       # EDA, SHAP, feature selection
├── 03_train_clear.py             # Train hybrid model on clear data
├── 04_encrypt_decrypt.py         # AES-GCM + CKKS encryption/decryption
├── 05_train_decrypted.py         # Retrain on decrypted data (with model encryption)
│
├── requirements.txt              # Dependencies
├── README.md                     # This file
│
├── data/                         # (not included, download separately)
│   ├── Patient_Info_Extended.csv
│   └── Patient-XXX/ (NIfTI files)
│
├── dataset_slices/               # Extracted PNG slices
├── dataset_slices_encrypted/     # Encrypted slices (.enc files)
├── dataset_slices_decrypted/     # Decrypted slices for training
│
├── models/                       # Saved models
│   ├── best_model.pth (encrypted after training)
│   └── best_model_encrypted.enc
│
└── results/                      # Figures and metrics
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── learning_curves.png
    └── shap_plots/
```

---

## 🖥️ Usage Examples

### Extract slices from NIfTI

```python
python 01_extract_slices.py
```

### Run SHAP analysis and feature selection

```python
python 02_dataset_balancing.py
```

### Train model on clear data

```python
python 03_train_clear.py
```

### Encrypt dataset and images

```python
python 04_encrypt_decrypt.py
# Follow prompts to enter encryption password
```

### Train on decrypted data (with automatic model encryption)

```python
python 05_train_decrypted.py
# Enter password when prompted (used for model encryption)
```

---

## 📈 Results Visualization

The training scripts automatically generate:
- Confusion matrix
- ROC curves (one-vs-rest) with AUC
- Learning curves (loss and accuracy over epochs)
- SHAP beeswarm and waterfall plots
- Feature importance bar charts

All figures are saved in the `results/` directory.

---

## ⚠️ Known Limitations

1. **Small dataset** – only 60 patients (though augmented via slice extraction and balancing)
2. **Image resolution** – reduced from 256×256 to 64×64 due to Colab GPU memory limits (13 GB RAM, 15 GB VRAM)
3. **Homomorphic encryption overhead** – training directly on encrypted images was computationally infeasible; we decrypt before training (still ensures privacy at rest and in transit)
4. **Google Colab session limits** – ~4 hours GPU per session; checkpoint system mitigates this

---

## 🔮 Future Work

- **Larger multi-center dataset** to improve generalization
- **Train directly on homomorphically encrypted data** using TenSEAL (requires more powerful hardware)
- **Longitudinal MRI data** to predict disease trajectory (not just current EDSS)
- **Deploy as a web API** with on-the-fly encryption for clinical use
- **Integrate with FHIR/HL7** standards for hospital information systems

---

## 👥 Authors

- **Francesco Alfonso Barlotti** – matr. 0522700022  
- **Salvatore Colucci** – matr. 0522700024  
- **Domenico Falco** – matr. 0522700023  

**Supervisors:**  
Prof. Stefano Cirillo, Prof. Vincenzo Deufemia  
*Università degli Studi di Salerno – Department of Computer Science*

---

## 📚 Citation

If you use this code or dataset in your research, please cite:

Also cite the original dataset:

```bibtex
@data{8bctsm8jz7_1,
  title = {Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation},
  author = {...},
  year = {2023},
  publisher = {Mendeley Data},
  doi = {10.17632/8bctsm8jz7.1}
}
```

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgments

- The MSLesSeg team for dataset annotations
- TenSEAL library for homomorphic encryption tools
- SHAP authors for model interpretability framework
- Google Colab for free GPU access

---
