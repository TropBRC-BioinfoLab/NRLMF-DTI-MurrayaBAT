##NRLMF-Guided Drug–Target Interaction Modeling for Murraya paniculata

This repository contains the implementation of a hybrid drug–target interaction (DTI) framework that integrates Neighborhood Regularized Logistic Matrix Factorization (NRLMF) with supervised machine learning for biologically informed negative sampling and compound–protein interaction prediction.

## 📌 Overview

Drug–target interaction (DTI) prediction plays a critical role in computational drug discovery. However, a fundamental challenge in DTI modeling is the extreme imbalance between known interactions and the vast number of unlabeled pairs, which are often incorrectly treated as negative samples.

To address this limitation, we propose a **NRLMF-guided negative sampling strategy**, where latent interaction scores derived from similarity-aware matrix factorization are used to identify biologically implausible compound–protein pairs.

This framework is applied to prioritize putative interactions between bioactive compounds of *Murraya paniculata* and brown adipose tissue (BAT)-associated proteins for anti-obesity research.

## 🔬 Key Contributions
- Introduces a **hybrid DTI modeling framework** combining:
  - Semi-supervised learning (NRLMF)
  - Supervised machine learning (RF, MLP, XGBoost, LR)
    
- Proposes **latent score-based negative sampling**:
  - Uses raw latent interaction scores (without sigmoid transformation)
  - Preserves latent space geometry for better discrimination
  
- Reduces **false-negative bias** in DTI datasets
  
- Demonstrates improved performance over random sampling on:
  - Internal dataset
  - Yamanishi (2008) benchmark datasets

## 🧠 Methodological Insight
Unlike conventional approaches that assume all unlabeled pairs are negative, this method:

- Models **drug–drug and target–target similarity**
- Learns **latent representations** of compounds and proteins
- Uses **latent interaction score (before sigmoid)**

- Ranks unlabeled pairs and selects:
  - **Lowest scores → candidate negatives**
  
This reflects **latent incompatibility** rather than naive absence of interaction.

---

## 🧪 Workflow
The overall pipeline consists of six stages:
1. Data acquisition (BindingDB, OMIM, GeneCards, UniProt)
2. Data preprocessing
3. Feature extraction:
   - Protein: AAC, AAIndex1, PAAC, ATC
   - Compound: ECFP, Morgan, MACCS, PubChem
4. NRLMF-based negative sampling
5. Supervised machine learning modeling
6. Prediction and biological interpretation
