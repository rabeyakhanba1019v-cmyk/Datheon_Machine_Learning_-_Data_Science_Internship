# Datheon Machine Learning & Data Science Internship

This repository contains my solutions for the **Datheon Machine Learning & Data Science Internship** technical tasks. The projects demonstrate fundamental machine learning concepts, including data preprocessing, exploratory data analysis (EDA), supervised learning, model evaluation, and visualization using Python and Scikit-Learn.

---

## Repository Structure

```
Datheon_Machine_Learning_&_Data_Science_Internship/
│
├── Task_1/
│   └── Task_1.ipynb
│
├── Task_2/
│   └── Task_2.ipynb
│
├── Task_3/
│   └── Task_3.ipynb
│
├── Task_4/
│   └── Task_4.ipynb
│
└── README.md
```

---

# Tasks

## Task 1 – Multi-Class Benchmark Classification (Iris Dataset)

### Objective

Build an end-to-end multi-class classification pipeline using the Iris dataset with Scikit-Learn.

### Dataset

- Iris Dataset
- 150 samples
- 4 numerical features
- 3 target classes

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

### Workflow

- Load dataset
- Exploratory Data Analysis (EDA)
- Pairplot visualization
- Data preprocessing
- Train/Test Split (80/20)
- Feature Scaling (StandardScaler)
- Logistic Regression
- Decision Tree Classifier
- Model Evaluation
- Confusion Matrix Heatmaps
- Decision Boundary Visualization
- Feature Importance Analysis
- Decision Tree Visualization

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

### Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|---------:|----------:|--------:|---------:|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

### Conclusion

Both Logistic Regression and Decision Tree achieved an accuracy of **93.33%** on the Iris dataset. The models successfully classified the three Iris species with high precision, recall, and F1-score. Since both models produced identical evaluation metrics, they demonstrated comparable classification performance on this benchmark dataset. The project highlights the complete machine learning workflow, including data preprocessing, feature scaling, model training, evaluation, and visualization.

---


## Task 2 – California Housing Price Prediction (Regression)

### Objective

Develop and compare multiple regression models to predict median house values using the California Housing dataset. The project includes data preprocessing, feature engineering, multicollinearity analysis, hyperparameter tuning, and model evaluation.

### Dataset

- California Housing Dataset
- 20,640 samples
- Numerical and categorical features
- Target: Median House Value

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Statsmodels

### Workflow

- Load California Housing dataset
- Exploratory Data Analysis (EDA)
- Missing value imputation
- Feature engineering
- Correlation heatmap
- Multicollinearity analysis (VIF)
- One-Hot Encoding for categorical features
- Feature scaling using StandardScaler
- Train/Test Split (80/20)
- Ordinary Least Squares (OLS) Regression
- Ridge Regression with GridSearchCV
- Lasso Regression with GridSearchCV
- Hyperparameter tuning
- Model comparison
- Predicted vs Actual visualization
- Residual distribution analysis

### Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### Results

| Model | RMSE | MAE | R² |
|-------|------:|------:|------:|
| OLS | 69,127.04 | 49,645.49 | 0.6353 |
| Lasso | 69,128.09 | 49,646.43 | 0.6353 |
| Ridge | 69,134.97 | 49,652.14 | 0.6353 |

**Best Hyperparameters**

- Ridge Alpha: **1**
- Lasso Alpha: **1**

### Conclusion

The Ordinary Least Squares (OLS) regression model achieved the best overall performance, with the lowest RMSE (69,127.04) and the highest R² score (0.6353). Ridge and Lasso regression produced nearly identical results, indicating that regularization had minimal impact on this dataset.

---

# Task 3 – Customer Churn Prediction

### Objective

Predict customer churn using ensemble machine learning techniques on the IBM Telco Customer Churn dataset.

### Algorithms

- Random Forest
- XGBoost

### Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)
- Matplotlib

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|---------:|----------:|-------:|---------:|---------:|
| Random Forest | 0.7736 | 0.5868 | 0.4973 | 0.5384 | 0.8197 |
| XGBoost | 0.7935 | 0.6224 | 0.5642 | 0.5919 | 0.8393 |

---

# Key Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Feature Scaling
- One-Hot Encoding
- SMOTE for Class Imbalance
- Machine Learning Classification
- Regression Analysis
- Ensemble Learning
- Hyperparameter Tuning
- Cross Validation
- Model Evaluation
- Data Visualization

---

# Libraries Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Statsmodels
- Matplotlib
- Seaborn
- Imbalanced-Learn

---

## Task 4

Solution available in **Task_4/**

---

## Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Author

Rabeya Khan Tasnuva
