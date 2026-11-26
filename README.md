# Datheon Machine Learning & Data Science Internship

This repository contains my solutions for the **Datheon Machine Learning & Data Science Internship** technical tasks. The projects demonstrate fundamental machine learning concepts, including data preprocessing, exploratory data analysis (EDA), supervised learning, model evaluation, visualization, and natural language processing (NLP) using Python and Scikit-Learn.

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

Both Logistic Regression and Decision Tree achieved an accuracy of **93.33%** on the Iris dataset. The models successfully classified the three Iris species with high precision, recall, and F1-score. Since both models produced identical evaluation metrics, they demonstrated comparable classification performance on this benchmark dataset.

---

# Task 2 – California Housing Price Prediction (Regression)

## Objective

Develop and compare multiple regression models to predict median house values using the California Housing dataset.

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
- One-Hot Encoding
- Feature Scaling
- Train/Test Split (80/20)
- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- Hyperparameter tuning using GridSearchCV
- Model comparison
- Prediction visualization
- Residual analysis

### Evaluation Metrics

- RMSE
- MAE
- R² Score

### Results

| Model | RMSE | MAE | R² |
|-------|------:|------:|------:|
| OLS | 69,127.04 | 49,645.49 | 0.6353 |
| Lasso | 69,128.09 | 49,646.43 | 0.6353 |
| Ridge | 69,134.97 | 49,652.14 | 0.6353 |

### Best Hyperparameters

- Ridge Alpha = **1**
- Lasso Alpha = **1**

### Conclusion

The OLS regression model achieved the best performance with the lowest RMSE and the highest R² score. Ridge and Lasso regression produced very similar results, indicating minimal benefit from regularization on this dataset.

---

# Task 3 – Customer Churn Prediction (Classification)

## Objective

Develop and compare ensemble machine learning models for customer churn prediction using the IBM Telco Customer Churn dataset.

### Dataset

- IBM Telco Customer Churn Dataset
- 7,043 customer records
- Numerical and categorical features
- Target: Customer Churn

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)
- Matplotlib

### Workflow

- Load dataset
- Handle missing values
- Encode categorical variables
- Feature scaling
- Train/Test Split (80/20)
- Apply SMOTE
- Train Random Forest
- Train XGBoost
- Perform 5-Fold Cross Validation
- Evaluate models
- Plot ROC Curves
- Plot Confusion Matrix
- Feature Importance Analysis

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|---------:|----------:|--------:|---------:|---------:|
| Random Forest | 0.7736 | 0.5868 | 0.4973 | 0.5384 | 0.8197 |
| XGBoost | 0.7935 | 0.6224 | 0.5642 | 0.5919 | 0.8393 |

### Conclusion

XGBoost outperformed Random Forest, achieving **79.35% Accuracy** and **0.8393 ROC-AUC**. The project demonstrates a complete classification workflow including preprocessing, class balancing, ensemble learning, cross-validation, and performance evaluation.

---

# Task 4 – Sentiment Analysis using NLP (Text Classification)

## Objective

Develop an end-to-end Natural Language Processing (NLP) pipeline to classify customer product and movie reviews into **positive** and **negative** sentiment classes using TF-IDF feature extraction and machine learning algorithms.

### Dataset

- Sentiment Reviews Dataset
- Product and Movie Reviews
- Positive and Negative Sentiment Labels
- Neutral reviews removed for binary classification

### Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- Matplotlib
- Seaborn
- Pickle

### Workflow

- Load sentiment review dataset
- Remove neutral reviews
- Convert text to lowercase
- Remove HTML tags
- Remove punctuation and numbers
- Remove stopwords
- Apply WordNet lemmatization
- Train/Test Split (80/20)
- TF-IDF Vectorization (`max_features=5000`)
- Train Multinomial Naive Bayes classifier
- Train Logistic Regression classifier
- Evaluate both models
- Generate Classification Reports
- Plot Confusion Matrices
- Compare model performance
- Build interactive sentiment prediction helper
- Save trained model and TF-IDF vectorizer

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
| Multinomial Naive Bayes | 0.5000 | 0.2500 | 0.5000 | 0.3333 |
| Logistic Regression | 0.5000 | 0.2500 | 0.5000 | 0.3333 |

### Conclusion

Both Multinomial Naive Bayes and Logistic Regression achieved identical performance on the provided sentiment dataset. Since the dataset contained only **10 reviews** (9 after removing the neutral review), the test set consisted of only **2 samples**, making the evaluation metrics highly sensitive to individual predictions. Despite the limited dataset size, the project successfully demonstrates the complete NLP workflow, including text preprocessing, TF-IDF feature extraction, machine learning model training, evaluation, interactive sentiment prediction, and model serialization using Pickle.

---

# Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- statsmodels
- xgboost
- imbalanced-learn

---

# Author

**Rabeya Khan Tasnuva**
