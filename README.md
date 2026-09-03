# 🛡️ ChurnGuard

## Customer Churn Prediction using Machine Learning

ChurnGuard is an interactive machine learning web application that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, tenure, and billing information.

The project uses Logistic Regression for binary classification and provides churn probability, customer risk assessment, prediction explanations, and model performance analysis.

---

## 🚀 Features

- 🔮 Customer churn prediction
- 📊 Churn probability and stay probability
- 🟢🟡🔴 Customer risk assessment
- 🧠 Prediction explanation using Logistic Regression coefficients
- 🔴 Factors pushing toward churn
- 🟢 Factors pushing toward staying
- 💼 Business-oriented retention recommendations
- 📈 ROC Curve
- 📊 ROC-AUC
- 🎯 Confusion Matrix
- 📋 Accuracy, Precision, Recall and F1 Score
- 📊 Feature influence analysis
- 🌐 Interactive Streamlit web application

---

## 🧠 Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Categorical Encoding
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
Streamlit Web Application
        ↓
Churn Prediction + Explanation
```

---

## 📊 Dataset 

This project uses the Telco Customer Churn dataset.

The dataset contains information about customers of a telecommunications company, including:

- Customer demographics
- Tenure
- Contract type
- Phone services
- Internet services
- Additional services
- Payment method
- Monthly charges
- Total charges
- Churn status

### Target Variable

`Churn`

```text
Yes → Customer churned
No  → Customer stayed
```

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify patterns related to customer churn.

The analysis includes:

- Dataset structure and information
- Missing-value checking
- Duplicate checking
- Categorical feature analysis
- Numerical feature analysis
- Churn distribution
- Feature relationships
- Logistic Regression coefficient analysis

---

## 🤖 Machine Learning Model

### Logistic Regression

ChurnGuard uses Logistic Regression for binary classification.

The model estimates the probability that a customer belongs to the churn class.

```text
P(Churn) = Probability that the customer will churn
```

The predicted probability is then used to determine the customer's risk level.

---

## ⚙️ Data Preprocessing

The application applies the same preprocessing approach used during model training.

### 1. Categorical Encoding

Categorical variables are converted into numerical features using one-hot encoding.

### 2. Feature Alignment

The input customer data is aligned with the exact feature columns used during model training.

### 3. Feature Scaling

Features are scaled using the saved StandardScaler.

### 4. Prediction

The processed data is passed to the trained Logistic Regression model.

---

## 📈 Model Evaluation

The model is evaluated using several classification metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Measures the percentage of overall correct predictions |
| Precision | Measures how many predicted churn customers actually churned |
| Recall | Measures how many actual churn customers were detected |
| F1 Score | Balances Precision and Recall |
| ROC-AUC | Measures how well the model separates churn from non-churn customers |

### ROC-AUC Result

The model achieved an ROC-AUC of approximately:

**0.862**

---

## 🧠 Model Explainability

ChurnGuard provides an explanation of why the model made a particular prediction.

The application calculates feature contributions using the customer's processed feature values and the Logistic Regression coefficients.

### Positive Contribution

A positive contribution pushes the prediction toward:

**Churn**

### Negative Contribution

A negative contribution pushes the prediction toward:

**Stay**

The application displays the strongest factors in two groups:

- 🔴 Factors pushing toward churn
- 🟢 Factors pushing toward staying

This makes the model's prediction easier to interpret.

---

## 💼 Business Use Case

Customer churn can lead to lost customers and potential revenue loss.

A churn prediction system can help businesses identify customers who may have a higher probability of leaving.

Example workflow:

```text
High Churn Risk
       ↓
Identify Customer
       ↓
Understand Contributing Factors
       ↓
Take Retention Action
```

Possible retention actions include:

- Customer support outreach
- Personalized offers
- Contract incentives
- Service improvements
- Customer satisfaction follow-up

The model provides predictions and insights. Final business decisions should consider additional customer and business information.

---

## 🌐 Web Application

ChurnGuard is built using Streamlit.

Users can enter customer information including:

- Gender
- Senior citizen status
- Tenure
- Partner
- Dependents
- Phone service
- Multiple lines
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming TV
- Streaming movies
- Contract
- Paperless billing
- Payment method
- Monthly charges
- Total charges

The application then produces:

```text
Customer Information
        ↓
Preprocessing 
        ↓
Logistic Regression
        ↓
Churn Probability
        ↓
Risk Level
        ↓
Prediction Explanation
        ↓
Business Recommendation
```

---

## 📊 Application Dashboard

### 🔮 Churn Prediction

The prediction dashboard provides:

- Customer input form
- Churn probability
- Stay probability
- Risk level
- Probability visualization
- Churn factors
- Stay factors
- Business recommendation

### 📈 Model Performance

The model performance dashboard provides:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Feature influence
- Model information

---

## 📂 Project Structure

```text
ChurnGuard/
│
├── app.py
├── eda.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
│
└── screenshots/
    └── churnguard.png
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## ▶️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ChurnGuard.git
```

### 2. Navigate to the project directory

```bash
cd ChurnGuard
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📸 Screenshots

Add screenshots of the application inside the `screenshots` folder.

Example:

```markdown
![ChurnGuard Application](screenshots/churnguard.png)
```

---

## 📚 Key Learning Outcomes

This project helped me practice:

- Data cleaning
- Exploratory Data Analysis
- Categorical feature encoding
- Feature scaling
- Train/test splitting
- Logistic Regression
- Classification metrics
- ROC curves
- ROC-AUC
- Confusion matrices
- Model interpretation
- Model serialization using Joblib
- Building an interactive machine learning application using Streamlit

---

## 🔮 Future Improvements

Potential future improvements include:

- Hyperparameter tuning
- Cross-validation
- Comparing multiple classification algorithms
- Probability calibration
- Experiment tracking
- Cloud deployment
- Model monitoring

---

## 👨‍💻 Author

**Yash Ohol**

Machine Learning Project  
Customer Churn Prediction

---

## ⭐ Project

If you found this project useful, consider giving the repository a star. 

