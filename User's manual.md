
## 📖 Overview
This project is a comprehensive Machine Learning pipeline designed to predict loan defaults. It serves as both a practical prediction tool and an educational comparison between a **custom-built Logistic Regression algorithm** (implemented from scratch using NumPy) and industry-standard models from **Scikit-Learn** (Logistic Regression, Random Forest, and Gradient Boosting).

The goal is to demonstrate the internal mechanics of logistic regression while benchmarking it against powerful ensemble methods.

## ⚙️ Prerequisites

To run this project, you will need a Python 3.x environment with the following dependencies installed:

- **Python 3.x**
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations and matrix operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-Learn**: For preprocessing, model benchmarking, and metrics.

### Installation Command
You can install all necessary libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
📂 Project StructureThe project is encapsulated in a single Jupyter Notebook (.ipynb), logically divided into the following sections:Imports & Setup: Loading libraries and suppressing warnings.Data Loading: Ingesting the Loan_Default.csv dataset.Exploratory Data Analysis (EDA):Analyzing data shape, info, and statistics.Visualizing the class imbalance in the Status column.Plotting correlation heatmaps.Data Preprocessing:Imputation: Handling missing values using SimpleImputer.Encoding: Converting categorical variables to numbers using LabelEncoder.Scaling: Standardizing features with StandardScaler (crucial for Gradient Descent).Model Implementation:LogisticRegressionScratch: A custom class implementing Sigmoid activation, Cost function, and Gradient Descent.Scikit-Learn wrappers for LR, Random Forest, and Gradient Boosting.Training & Evaluation: A systematic training loop that collects metrics (Accuracy, Precision, Recall, F1, ROC-AUC).Visualization:Cost history plots for the custom model.Confusion matrices.ROC Curve overlays.🚀 How to Use1. Data SetupEnsure you have the dataset file named Loan_Default.csv.Place the file in the same directory as the notebook, ORUpdate the file path in the main() function or data loading cell:Python# In the main() function or global scope
df = pd.read_csv("path/to/Loan_Default.csv")
2. Running the ProgramOpen the notebook in Jupyter Lab, Jupyter Notebook, or Google Colab.The notebook is designed to be run sequentially. However, the entire workflow is orchestrated by the main() function at the end of the script.To execute the full pipeline:Run the final cell containing the main execution block:Pythonif __name__ == "__main__":
    main()
3. Verifying Custom LogicIncluded is a helper function test_on_small_data() to verify that the custom Logistic Regression math works correctly on a simple, synthetic dataset before running it on the real data.Python# Run this cell to sanity check the custom model
test_on_small_data()
📊 Models ImplementedModelTypeDescriptionLogistic Regression (Custom)ClassificationBuilt from scratch using Gradient Descent to minimize Binary Cross-Entropy loss.Logistic Regression (Sklearn)ClassificationStandard implementation used as a baseline for correctness.Random ForestEnsembleBagging method using decision trees; robust to overfitting.Gradient BoostingEnsembleBoosting method that builds trees sequentially to correct errors.📈 Expected OutputsWhen you run the program, it will generate:Text Output:Data shape and type information.Class distribution counts.A comparison table of metrics (Accuracy, Precision, Recall, F1, ROC-AUC).Top 10 most important features (from the Random Forest model).Visualizations:Distribution Plot: Shows the ratio of defaults vs. non-defaults.Training Cost Curve: Shows how the custom model's error decreased over iterations.Confusion Matrices: 2x2 grids showing True Positives, False Positives, etc., for all models.ROC Curves: A graph comparing the True Positive Rate vs. False Positive Rate for all models.🛠️ ConfigurationYou can tune the custom model hyperparameters in the main() function:Python# Example: Adjusting learning rate and iterations
custom_lr = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
📜 LicenseThis project is open-source and available for educational and personal use.
