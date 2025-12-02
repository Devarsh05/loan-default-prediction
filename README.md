# Loan Default Prediction: Robust Modeling and Leakage Mitigation

## Project Overview

This project implements a comprehensive machine learning pipeline to predict loan default (`Status=1`) using a proprietary financial dataset. The primary technical challenge involved identifying and resolving severe **Target Leakage** present in the source data, which led to artificially perfect model scores.

The solution includes:
1.  A fully built **Logistic Regression (LR) model from scratch** using NumPy, fulfilling the core academic requirement.
2.  A robust **data engineering pipeline** designed to prevent both methodological and feature-level data leakage.
3.  Evaluation and comparison of six predictive models, including state-of-the-art ensemble methods.

## Key Features and Goals

- **Scratch Implementation:** Built **Logistic Regression** using Gradient Descent from first principles.
- **Leakage Mitigation:** Successfully identified and **dropped eight highly correlated proxy features** (e.g., `rate_of_interest`, `Upfront_charges`, `credit_type`) to ensure the final models yield trustworthy, non-perfect scores.
- **Model Comparison:** Trained and evaluated six distinct models (LR Scratch, LR Sklearn, RF, GB, XGBoost, SVM).
- **Robust Preprocessing:** Implemented a non-leaky pipeline using train-set-only fitting for all imputation and scaling steps.

## 📈 Methodology and Pipeline Overview

The project follows a standard machine learning workflow, with a critical emphasis on separating the data for honest evaluation.

1.  **Data Preparation:** Load data, sample 5,000 records, and aggressively drop target-leaking features.
2.  **Data Split:** The data is split **first** (Train/Validation/Test, stratified) before any transformations.
3.  **Preprocessing:** Imputation, One-Hot Encoding, and Scaling are **fitted exclusively on the training data** to prevent leakage.
4.  **Training:** Six models are trained on the processed training set.
5.  **Evaluation:** Models are evaluated on the unseen **Validation Set** using robust metrics suitable for imbalanced data.

### Scratch Model: Logistic Regression

Our custom-built model uses **Batch Gradient Descent** to optimize the weights and bias by minimizing the **Binary Cross-Entropy Loss**.

The model calculates a linear combination of features, passes it through the **Sigmoid function** 

[Image of Sigmoid function]
 to obtain a probability, and then uses the backpropagation principle to adjust parameters iteratively.

## 📊 Final Model Performance Summary

After resolving the data leakage, the models show realistic predictive power. Performance is measured using the **ROC-AUC** and **F1-Score** (best for imbalanced classification).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.8653 | **0.9151** | 0.5132 | 0.6576 | **0.8869** |
| **Gradient Boosting** | **0.8760** | 0.8810 | 0.5873 | **0.7048** | 0.8722 |
| **XGBoost** | 0.8680 | 0.8261 | **0.6032** | 0.6972 | 0.8672 |
| SVM | 0.7680 | 0.5904 | 0.2593 | 0.3603 | 0.7734 |
| LR (Scratch) | 0.7733 | 0.7021 | 0.1746 | 0.2797 | 0.7358 |

**Conclusion:** The **Gradient Boosting** model is the strongest performer, achieving the highest F1-Score, indicating the best balance between correctly identifying defaults (Recall) and minimizing false alarms (Precision).

## ⚠️ Target Leakage Mitigation

The final, stable model performance was only achieved after an extensive debugging process that targeted data leakage.

| Leak Feature Dropped | Rationale |
| :--- | :--- |
| `Credit_Worthiness`, `credit_type` | Highly correlated categorical features likely derived from the final status. |
| `loan_limit`, `approv_in_adv` | Features suggesting a post-determination outcome rather than a pre-application input. |
| `Upfront_charges` | A fee that may only be applied upon a successful/finalized outcome. |
| `Interest_rate_spread`, `rate_of_interest` | Numerical values derived from an assessment process that may have included the final risk status. |

## 💻 Code Structure

The main logic resides in the Python script/notebook and is organized by functionality:

| Component | Description |
| :--- | :--- |
| **Imports** | Setup for NumPy, Pandas, and all necessary Scikit-learn models/tools. |
| **`load_and_preprocess_data`** | The data engineering core. Handles splits, imputation, scaling, and OHE column alignment. |
| **`train_model`** | The scratch implementation of Logistic Regression (Gradient Descent loop). |
| **`sigmoid`, `compute_loss`, `compute_gradients`** | Auxiliary functions forming the math foundation of the scratch model. |
| **Main Execution Loop** | Trains the six models sequentially and generates the final comparison summary. |

## Setup and Prerequisites

To run this project, you need Python and the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
