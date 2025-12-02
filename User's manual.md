# 📘 Loan Default Prediction System — Manual

## 📖 Overview
This project is a complete **Machine Learning pipeline** for predicting **loan defaults**.  
It serves two main purposes:

1. **Educational** — Demonstrates how Logistic Regression works internally by implementing it from scratch using **NumPy** with custom Gradient Descent.
2. **Practical** — Compares the custom model with powerful, industry-standard **Scikit-Learn** algorithms:
   - Logistic Regression (Sklearn)
   - Random Forest
   - Gradient Boosting

This project helps you understand:
- How logistic regression works mathematically  
- How it performs relative to ensemble methods  
- How a full ML workflow is built, from EDA → Preprocessing → Training → Evaluation → Visualization  

---

## ⚙️ Prerequisites

You need **Python 3.x** and the following libraries:

- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-Learn**

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📂 Project Structure

This project is provided as a single **Jupyter Notebook (`.ipynb`)**.  
The notebook is organized as follows:

---

### 1. **Imports & Setup**
- Load required libraries  
- Suppress warnings for cleaner output  

---

### 2. **Data Loading**
- Reads the dataset: `Loan_Default.csv`

---

### 3. **Exploratory Data Analysis (EDA)**
- Inspect dataset shape, info, and statistics  
- Visualize the **class imbalance** in the `Status` column  
- Plot **correlation heatmaps** to analyze relationships  

---

### 4. **Data Preprocessing**
- **Missing value imputation** using `SimpleImputer`  
- **Categorical encoding** with `LabelEncoder`  
- **Feature scaling** using `StandardScaler`  
  (Important for Gradient Descent to converge properly)

---

### 5. **Model Implementation**
Includes:

#### ✔️ **Custom Logistic Regression (`LogisticRegressionScratch`)**
Implements:
- Sigmoid activation  
- Binary cross-entropy cost function  
- Gradient Descent  
- Cost history logging  

#### ✔️ **Sklearn Models**
- Logistic Regression  
- Random Forest  
- Gradient Boosting  

---

### 6. **Training & Evaluation**
The notebook:
- Trains each model  
- Collects performance metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - ROC-AUC  
- Displays results in a comparison table  

---

### 7. **Visualization**
Generated plots include:
- **Cost Curve** for custom logistic regression  
- **Confusion Matrices** for all models  
- **ROC Curves** comparing all models  

---

## 🚀 How to Use

### 1. Prepare the Dataset
Ensure the dataset file is named:

```
Loan_Default.csv
```

Place it in the same directory as the notebook.  
Or modify the path as needed:

```python
df = pd.read_csv("path/to/Loan_Default.csv")
```

---

### 2. Running the Notebook

Open in:
- Jupyter Notebook  
- Jupyter Lab  
- Google Colab  

Run the notebook **top to bottom**, or simply run the main execution block:

```python
if __name__ == "__main__":
    main()
```

This will:
- Load data  
- Preprocess  
- Train all models  
- Compute metrics  
- Generate all plots  

---

### 3. Verify Custom Logistic Regression Logic

You can test the scratch implementation on a small synthetic dataset before training on the real loan data:

```python
# Sanity check
test_on_small_data()
```

---

## 📊 Models Included

| Model | Type | Description |
|-------|------|-------------|
| **Logistic Regression (Custom)** | Classification | Built from scratch using Gradient Descent + BCE loss |
| **Logistic Regression (Sklearn)** | Classification | Standard benchmark model |
| **Random Forest** | Ensemble | Bagging-based tree ensemble |
| **Gradient Boosting** | Ensemble | Sequential boosting of decision trees |

---

## 📈 Expected Outputs

When running the project, you will see:

### ✔️ **Text Output**
- Data shape & type summary  
- Class distribution (default vs. non-default)  
- Model performance table  
- Top 10 most important features (from Random Forest)

### ✔️ **Visualizations**
- **Distribution Plot** (class imbalance)  
- **Training Cost Curve** (for custom LR)  
- **Confusion Matrices** (2×2 grids)  
- **ROC Curves** for all models  

---

## 🛠️ Configuration

You can tune the custom logistic regression model:

```python
custom_lr = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iterations=1000
)
```

Feel free to modify:
- Learning rate  
- Number of iterations  
- Batch size (if implemented)  
- Regularization (if extended)  

---

## 📜 License
This project is **open-source** and available for **educational** and **personal** use.
