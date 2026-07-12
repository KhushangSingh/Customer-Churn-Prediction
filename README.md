# Customer Churn Prediction for a Subscription Streaming Service

**Author:** Khushang Singh  
**Application No.:** IN26013157  
**Domain:** Media & Subscription Business Analytics  

## Project Overview
A streaming platform loses a meaningful share of subscribers every month. The objective of this project is to identify in advance which customers are likely to cancel their subscription (churn). By building a rigorous, well-validated predictive model, the marketing team can efficiently target retention offers and allocate budget based on the findings.

## Dataset
* **Name:** Telco Customer Churn (IBM Sample Dataset)
* **Source:** Kaggle
* **URL:** [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Methodology
The project treats the Jupyter Notebook like a well-structured software pipeline, using Scikit-learn for all modeling steps. Key phases include:

* **Exploratory Data Analysis (EDA) & Preprocessing:** Handling missing values, mapping binary variables, one-hot encoding categorical features, and standardizing numerical features.
* **Feature Selection:** Applying Tree-Based Feature Importance (Random Forest) and Correlation Filtering to reduce dimensionality, prevent overfitting to noise, and optimize training speed.
* **Model Comparison:** Training and evaluating Random Forest and Support Vector Machine (Linear SVM) algorithms.
* **Cross-Validation:** Utilizing 5-fold cross-validation on both the full feature set and the reduced feature set to ensure results are not driven by a lucky train/test split.
* **Hyperparameter Tuning:** Optimizing the best-performing model (Random Forest) using `GridSearchCV`.
* **Final Evaluation:** Assessing the final model comprehensively using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.

## Business Recommendations
The Tuned Random Forest on the reduced feature set balances predictive performance with business interpretability. Based on the model's findings, the following actionable retention strategies are recommended:

1. **Targeted Friction Reduction in the First 6 Months:** `Tenure` is the strongest predictor of churn. We should trigger automated, in-app onboarding tutorials and prompts for users in their first 3 to 6 months to deepen engagement.
2. **Revamp the Month-to-Month Payment Flow:** Customers on `Month-to-Month` contracts are high-risk. We should embed a seamless one-click upgrade path to a 1-year contract within the user dashboard to convert them into long-term subscribers.
3. **Address the Electronic Check Pain Point:** `PaymentMethod_Electronic check` is a significant churn indicator. The team should audit this payment gateway for UX friction and offer incentives to migrate users to automated credit card billing.

## How to Run the Project
1. Clone this repository to your local machine.
2. Install the required dependencies using the terminal command: `pip install -r requirements.txt`. (Note: The notebook also includes a pip cell at the top for automated installation).
3. Open the Jupyter Notebook (`.ipynb` file).
4. Run the notebook sequentially from top to bottom. No manual intervention is required; the dataset is automatically fetched from a public raw URL during runtime.
