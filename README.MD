Customer Churn Prediction using Machine Learning

Project Overview

Customer churn prediction is a machine learning project that identifies customers who are likely to discontinue a service. The goal is to help businesses understand churn patterns and develop data-driven customer retention strategies.

Objectives

- Analyze customer demographics, services, contracts, and billing information.
- Perform data cleaning and preprocessing.
- Train and compare multiple machine learning models.
- Evaluate model performance using classification metrics.
- Identify important features associated with churn predictions.
- Generate actionable business recommendations.

Dataset

- Dataset: IBM Telco Customer Churn
- Records: 7,043 customers
- Features: 19 input features after removing the customer ID and target column
- Target: Churn ("Yes" / "No")
- Class distribution: 1,869 customers churned and 5,174 did not churn.

Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Joblib

Methodology

1. Exploratory Data Analysis: Examined dataset structure, missing values, and churn distribution.
2. Data Cleaning: Converted "TotalCharges" to numeric format and handled missing values through pipeline-based imputation.
3. Preprocessing: Applied median imputation and standardization to numerical features, and most-frequent imputation and one-hot encoding to categorical features.
4. Train-Test Split: Split the dataset into training and testing sets using stratification.
5. Model Training: Compared Logistic Regression, Balanced Logistic Regression, Tuned Random Forest, Gradient Boosting, and XGBoost.
6. Model Evaluation: Used accuracy, precision, recall, F1-score, and ROC-AUC, along with confusion matrices and precision-recall analysis.
7. Feature Importance: Examined the features most influential in the Random Forest model's predictions.

Model Comparison

Five models were compared using five-fold stratified cross-validation on the training data.

Model| F1-score| ROC-AUC
Logistic Regression| 0.592| 0.846
Balanced Logistic Regression| 0.628| 0.846
Tuned Random Forest| 0.631| 0.845
Gradient Boosting| 0.587| 0.849
XGBoost| 0.595| 0.849

Selected model: Tuned Random Forest, based on its highest mean cross-validation F1-score of 0.631. Balanced Logistic Regression is an alternative when higher churn recall is the priority.

Key Findings

The initial Random Forest feature-importance analysis highlighted the following influential features:

- Month-to-month contract
- Customer tenure
- Total charges
- Monthly charges
- Two-year contract
- Online security subscription status

These features indicate useful areas for further investigation into customer retention. Feature importance describes model reliance, not causation.

Business Recommendations

- Investigate why month-to-month customers may be more likely to churn.
- Consider retention offers for customers showing high predicted churn risk.
- Review onboarding and engagement strategies for newer customers.
- Evaluate whether suitable contract options and service bundles improve retention.
- Track churn rate, retention campaign conversion, and customer lifetime value.

These recommendations should be validated against business data and controlled retention experiments before implementation.

Project Structure

Customer_churn_prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── customer_churn_model.pkl
├── reports/
│   └── feature_importance.csv
├── 01_churn_analysis.ipynb
└── README.md

Limitations

- The model uses historical customer data and may not generalize to every business.
- The churn class is imbalanced, so accuracy alone is insufficient.
- Cross-validation results are estimates; final performance should be reported using the held-out test set.
- Feature importance does not establish causal relationships.

Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn analysis, including exploratory analysis, preprocessing, model comparison, evaluation, and business interpretation. It illustrates how predictive analytics can support customer retention decisions.

How to Run the Project

1. Clone the repository

git clone https://github.com/sneha20052005/Customer-churn-prediction.git
cd Customer-churn-prediction

2. Install the dependencies

pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter

3. Add the dataset

Obtain the IBM Telco Customer Churn dataset from its original source, subject to its applicable terms. Place the CSV at:

"data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

4. Run the notebook

Open "01_churn_analysis.ipynb" in Jupyter Notebook or VS Code and run the cells in order.

Dataset

This project uses the IBM Telco Customer Churn dataset, containing 7,043 customer records.

- Dataset source: "Kaggle — Telco Customer Churn" (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Download the CSV and place it in the "data/" folder.
- Expected filename: "WA_Fn-UseC_-Telco-Customer-Churn.csv"
- The raw dataset is not included in this repository.

Please follow the dataset source's applicable terms of use.
