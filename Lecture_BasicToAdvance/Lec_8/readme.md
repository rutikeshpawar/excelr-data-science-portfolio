# 🏦 Loan Approval Prediction System

A Machine Learning project that predicts whether a loan application is likely to be **Approved or Rejected** based on applicant, financial, employment, loan, and credit-related information.

The project compares multiple classification algorithms and deploys the best-performing model using **Streamlit**.

---

## 📌 Project Overview

Loan approval is an important decision in the banking and financial sector. Traditionally, loan applications are evaluated using multiple applicant and financial factors.

In this project, Machine Learning is used to analyze historical loan application data and build a classification model that predicts the loan approval status.

The project follows a complete Machine Learning workflow:

**Data Loading → Data Understanding → Data Cleaning → Exploratory Data Analysis → Encoding → Train-Test Split → Feature Scaling → Model Building → Model Evaluation → Model Comparison → Feature Importance → Model Saving → Streamlit Deployment**

---

## 🎯 Objective

The main objective of this project is to:

- Analyze loan application data.
- Understand the factors associated with loan approval.
- Preprocess categorical and numerical data.
- Build classification models.
- Compare Logistic Regression, KNN, and Random Forest.
- Select the best-performing model.
- Save the trained model for deployment.
- Build an interactive Streamlit application for real-time predictions.

---

## 📂 Dataset

The project uses a loan application dataset containing applicant, employment, loan, and credit information.

### Features

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_gender` | Applicant's gender |
| `person_education` | Applicant's education level |
| `person_income` | Applicant's annual income |
| `person_emp_exp` | Employment experience |
| `person_home_ownership` | Home ownership status |
| `loan_amnt` | Requested loan amount |
| `loan_intent` | Purpose of the loan |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a proportion of income |
| `cb_person_cred_hist_length` | Length of credit history |
| `credit_score` | Applicant's credit score |
| `previous_loan_defaults_on_file` | Previous loan default information |

### Target Variable

`loan_status`

```text
0 → Loan Rejected
1 → Loan Approved
````
---

# 🔄 Machine Learning Workflow

## 1. Data Loading

The loan dataset is loaded using Pandas and initially inspected to understand its structure, columns, data types, and dimensions.

---

## 2. Data Cleaning

The dataset is checked for:

* Missing values
* Duplicate records
* Incorrect data types
* Basic statistical information

Duplicate records are removed before model training.

---

## 3. Exploratory Data Analysis

Exploratory analysis is performed to understand the distribution of loan approval status and the characteristics of the dataset.

Visualizations are created using Matplotlib and Seaborn.

---

## 4. Feature and Target Separation

The dataset is divided into:

### Independent Variables

All applicant, employment, loan, and credit-related features.

### Dependent Variable

`loan_status`

---

## 5. Categorical Encoding

Machine Learning models require numerical input.

The following categorical variables are converted into numerical values using `LabelEncoder`:

```text
person_gender
person_education
person_home_ownership
loan_intent
previous_loan_defaults_on_file
```

The encoders are saved separately so that the Streamlit application can apply the **same transformations** to new user inputs.

---

## 6. Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

A fixed `random_state=42` is used to make the experiment reproducible.

Stratified splitting is used to maintain the distribution of the target classes.

---

## 7. Feature Scaling

`StandardScaler` is applied for models where feature scaling is beneficial, particularly:

* Logistic Regression
* KNN

Random Forest is trained using the encoded features without standard scaling.

---

# 🤖 Machine Learning Models

Three classification algorithms were evaluated.

## 1. Logistic Regression

Logistic Regression was implemented as a baseline classification model.

It achieved approximately:

```text
82.53% Accuracy
```

---

## 2. K-Nearest Neighbors (KNN)

KNN was implemented using scaled features.

The model was configured with:

```text
n_neighbors = 5
weights = distance
```

It was evaluated using the testing dataset.

---

## 3. Random Forest

Random Forest was selected as the final model because it achieved the strongest performance among the evaluated models.

Configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

### Final Accuracy

```text
92.61%
```

---

# 📊 Model Comparison

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression |    ~82.53% |
| KNN                 |  Evaluated |
| Random Forest       | **92.61%** |

Based on the evaluation results, **Random Forest was selected as the final model**.

---

# 📈 Model Evaluation

The Random Forest model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The classification report provides detailed performance for both:

```text
0 → Rejected
1 → Approved
```

---

# 🔍 Feature Importance

Random Forest provides feature importance values that help understand which variables contributed most to the model's predictions.

Feature importance was calculated using:

```python
rf.feature_importances_
```

This helps provide additional insight into the factors influencing loan approval predictions.

---

# 🧪 Model Validation

As an additional validation step, an actual approved loan record from the dataset was passed through the trained Random Forest model.

The model correctly predicted:

```text
Actual Status: 1
Model Prediction: 1
Approval Probability: ~93%
```

This confirmed that the saved model was functioning correctly on an approved example from the dataset.

---

# 💾 Saved Model Files

The trained Machine Learning components are saved using Joblib.

### Random Forest Model

```text
random_forest_model.pkl
```

This file contains the trained Random Forest classifier.

### Encoders

```text
encoders.pkl
```

This file contains the LabelEncoder objects used for categorical variables.

Saving the encoders ensures that Streamlit converts user inputs using the same mappings used during model training.

---

# 🌐 Streamlit Application

The trained model is deployed through a Streamlit application.

The application allows users to enter:

### Applicant Information

* Age
* Gender
* Education
* Annual Income
* Employment Experience
* Home Ownership

### Loan Information

* Loan Amount
* Loan Intent
* Interest Rate
* Loan Percentage of Income
* Credit History Length

### Credit Information

* Credit Score
* Previous Loan Default

The application then predicts:

```text
✅ LOAN APPROVED
```

or

```text
❌ LOAN REJECTED
```

It also displays the estimated approval probability.

---

# 🖥️ Running the Streamlit Application

Open the terminal in the project folder and run:

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 📁 Project Structure

```text
8_src/
│
├── Lec_8_KNN.ipynb
├── app.py
├── random_forest_model.pkl
├── encoders.pkl
├── loan_data.csv
├── DataDetails.txt
├── requirements.txt
├── Dockerfile
└── readme.md
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

---

# 🧠 Machine Learning Concepts Used

This project demonstrates:

* Data preprocessing
* Exploratory Data Analysis
* Categorical Encoding
* Train-Test Split
* Feature Scaling
* Classification
* Logistic Regression
* K-Nearest Neighbors
* Random Forest
* Model Evaluation
* Confusion Matrix
* Feature Importance
* Model Serialization
* Streamlit Deployment

---

# 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Advanced ensemble models
* Better handling of class imbalance
* Explainable AI techniques
* Improved UI/UX
* Cloud deployment
* Model monitoring
* Automated retraining pipeline

---

# ⚠️ Disclaimer

This project is created for **educational and demonstration purposes**.

The prediction produced by the application should not be considered an actual financial or lending decision.

---

# 👨‍💻 Project Summary

This project demonstrates an end-to-end Machine Learning workflow for loan approval prediction, starting from raw data analysis and preprocessing to model development, evaluation, serialization, and deployment through Streamlit.

Among the tested models, **Random Forest achieved the best performance with approximately 92.61% accuracy** and was therefore selected as the final prediction model.

The project demonstrates how a trained Machine Learning model can be integrated into an interactive application where users can enter loan information and receive a real-time prediction.
