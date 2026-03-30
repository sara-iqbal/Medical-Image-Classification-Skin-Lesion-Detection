# Medical Image Classification: Skin Lesion Detection

### CNN vs. Vision Transformer Comparative Analysis

## Overview

This project builds and compares two distinct deep learning architectures—a Convolutional Neural Network (MobileNetV2) and a Vision Transformer (ViT-Tiny)—for classifying skin lesion images into 7 diagnostic categories. The objective is to evaluate how traditional local feature extraction compares to global attention-based models in a clinical decision support context.

**[Link to Live Dashboard]([https://www.google.com/search?q=%23](https://sara-iqbal.github.io/Medical-Image-Classification-Skin-Lesion-Detection/))** | **[Link to Source Code]([https://www.google.com/search?q=%](https://drive.google.com/file/d/1HzdWPQL2c2Ex5dE1wyiWzf_USDEV-60v/view?usp=sharing)23)**
*(Update these links with your actual URLs)*

## The Dataset

The model is trained and evaluated on the **HAM10000** dataset, a benchmark collection of dermatoscopic images.

**Diagnostic Classes:**

  * MEL: Melanoma (Malignant)
  * NV: Melanocytic Nevi (Benign)
  * BCC: Basal Cell Carcinoma (Malignant)
  * AKIEC: Actinic Keratosis (Pre-malignant)
  * BKL: Benign Keratosis (Benign)
  * DF: Dermatofibroma (Benign)
  * VASC: Vascular Lesion (Benign)

## Technical Architecture

### 1\. Convolutional Neural Network (CNN)

  * **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
  * **Architecture:** Utilizes sliding convolutional filters to detect local textures, edges, and lesion borders.
  * **Custom Head:** Modified classifier with Dropout layers for regularization.

### 2\. Vision Transformer (ViT)

  * **Base Model:** ViT-Tiny (Patch 16, 224x224).
  * **Architecture:** Treats images as sequences of patches. Utilizes Self-Attention mechanisms to establish global context and capture structural symmetry across the entire lesion.

### Key Engineering Features

  * **Handling Class Imbalance:** The dataset features a severe **6.7:1 imbalance**, heavily skewed toward benign Melanocytic Nevi (NV). To prevent the model from biased guessing, **Weighted Cross-Entropy Loss** was implemented, applying higher penalties for misclassifying rare and dangerous classes like Melanoma.
  * **Memory Optimization:** Implemented a highly efficient PyTorch Dataset pipeline that streams images directly from the HuggingFace Apache Arrow cache, preventing system RAM crashes during training.
  * **Training Protocol:** Both models were trained for 10 epochs using a Cosine Annealing Learning Rate Scheduler to ensure smooth convergence.

## Results Comparison

The CNN slightly outperformed the ViT-Tiny in this specific configuration, likely due to the CNN's strong inductive biases being better suited for smaller datasets, whereas Vision Transformers typically require significantly more data to generalize effectively.

| Metric | CNN (MobileNetV2) | ViT (ViT-Tiny) |
| :--- | :--- | :--- |
| **Accuracy** | 83.2% | 81.7% |
| **F1 Macro** | 0.814 | 0.74 |
| **ROC-AUC (OvR)** | 0.912 | 0.84 |

## Repository Structure

  * `medical_image_classification.ipynb`: The main notebook containing the data pipeline, model training, evaluation, and visualizations.
  * `medical_cv_data.json`: The exported evaluation metrics, per-class F1 scores, and confusion matrix data used for the dashboard.

