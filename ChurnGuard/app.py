import pandas as pd 
import joblib

model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")

print("Model loaded successfully!")
print("Number of features:", len(model_columns))
print(model_columns)


import streamlit as st
import pandas as pd
import joblib


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")


# ============================================================
# LOAD DATASET FOR MODEL PERFORMANCE
# ============================================================

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# PREPARE DATASET
# ============================================================

df = df.drop(columns=["customerID"], errors="ignore")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()


# Convert SeniorCitizen into Yes / No
if "SeniorCitizen" in df.columns:

    df["SeniorCitizen"] = df["SeniorCitizen"].map({
        0: "No",
        1: "Yes"
    })


# Target
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn"])


# ============================================================
# ENCODE DATASET
# ============================================================

categorical_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# Make columns exactly the same as training
X_encoded = X_encoded.reindex(
    columns=model_columns,
    fill_value=0
)


# Scale
X_scaled = scaler.transform(X_encoded)


# Predictions for evaluation
y_pred = model.predict(X_scaled)

y_prob = model.predict_proba(X_scaled)[:, 1]


# ============================================================
# MODEL METRICS
# ============================================================

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y,
    y_prob
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #0b1120;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #111827, #172554);
    border: 1px solid #26334d;
    border-radius: 22px;
    padding: 38px 42px;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 8px;
}

.section-title {
    color: white;
    font-size: 23px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 18px;
}

.result-card {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-number {
    color: white;
    font-size: 46px;
    font-weight: 800;
    margin: 12px 0;
}

.result-label {
    color: #cbd5e1;
    font-size: 17px;
}

.factor-card {
    background: #111827;
    border: 1px solid #26334d;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 10px;
}

.factor-name {
    color: #e2e8f0;
    font-weight: 600;
}

.factor-value {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
}

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #26334d;
    padding: 20px;
    border-radius: 16px;
}

[data-testid="stSidebar"] {
    background: #080d18;
}

.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ ChurnGuard")

    st.markdown("### Customer Intelligence")

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Problem:** Binary Classification")
    st.write("**Target:** Customer Churn")
    st.write("**Output:** Churn Probability")

    st.markdown("---")

    st.markdown("### 📊 Pipeline")

    st.write("""
    Customer Data
    ↓
    One-Hot Encoding
    ↓
    Feature Alignment
    ↓
    Standard Scaling
    ↓
    Logistic Regression
    ↓
    Churn Probability
    """)

    st.markdown("---")

    st.caption("ChurnGuard • Machine Learning Project")


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🛡️ ChurnGuard
    </div>

    <div class="hero-subtitle">
        Predict customer churn before it happens.<br>
        Analyze customer behavior using machine learning.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

prediction_tab, performance_tab = st.tabs([
    "🔮 Churn Prediction",
    "📊 Model Performance"
])


# ============================================================
# PREDICTION TAB
# ============================================================

with prediction_tab:

    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    st.markdown(
        '<div class="section-title">👤 Customer Profile</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )


    with col2:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )


    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    # ========================================================
    # SERVICES
    # ========================================================

    st.markdown(
        '<div class="section-title">📡 Services</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    # ========================================================
    # BILLING
    # ========================================================

    st.markdown(
        '<div class="section-title">💳 Billing Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )


    with col2:

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "🔮 ANALYZE CUSTOMER",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        # ----------------------------------------------------
        # CREATE CUSTOMER DATA
        # ----------------------------------------------------

        customer = pd.DataFrame({

            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]

        })


        # ----------------------------------------------------
        # ENCODE
        # ----------------------------------------------------

        customer_encoded = pd.get_dummies(
            customer,
            columns=categorical_columns,
            drop_first=True
        )


        # ----------------------------------------------------
        # ALIGN FEATURES
        # ----------------------------------------------------

        customer_encoded = customer_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        customer_scaled = scaler.transform(
            customer_encoded
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            customer_scaled
        )[0]

        probability = model.predict_proba(
            customer_scaled
        )[0][1]


        # ====================================================
        # RISK LEVEL
        # ====================================================

        if probability >= 0.5:

            risk = "HIGH RISK"
            icon = "🔴"
            message = "This customer is predicted to churn."

        elif probability >= 0.3:

            risk = "MEDIUM RISK"
            icon = "🟡"
            message = "This customer has moderate churn risk."

        else:

            risk = "LOW RISK"
            icon = "🟢"
            message = "This customer is predicted to stay."


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="result-card">

            <div class="result-title">
                Customer Risk Assessment
            </div>

            <div class="result-number">
                {icon} {risk}
            </div>

            <div class="result-label">
                {message}
            </div>

        </div>
        """, unsafe_allow_html=True)


        # ====================================================
        # PROBABILITY METRICS
        # ====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.1f}%"
            )


        with col2:

            st.metric(
                "Stay Probability",
                f"{(1 - probability) * 100:.1f}%"
            )


        with col3:

            st.metric(
                "Risk Level",
                risk
            )


        st.markdown("### Churn Probability")

        st.progress(
            float(probability)
        )

        st.caption(
            f"{probability * 100:.1f}% predicted probability of churn"
        )


        # ====================================================
        # INDIVIDUAL EXPLANATION
        # ====================================================

        st.markdown(
            '<div class="section-title">🧠 Why did the model predict this?</div>',
            unsafe_allow_html=True
        )


        coefficients = model.coef_[0]

        contributions = (
            customer_scaled[0] * coefficients
        )


        explanation = pd.DataFrame({

            "Feature": model_columns,

            "Contribution": contributions

        })


        # Positive = pushes toward churn
        churn_factors = (
            explanation[
                explanation["Contribution"] > 0
            ]
            .sort_values(
                "Contribution",
                ascending=False
            )
            .head(5)
        )


        # Negative = pushes toward staying
        stay_factors = (
            explanation[
                explanation["Contribution"] < 0
            ]
            .sort_values(
                "Contribution",
                ascending=True
            )
            .head(5)
        )


        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # CHURN FACTORS
        # ----------------------------------------------------

        with col1:

            st.markdown(
                "### 🔴 Factors pushing toward churn"
            )

            if churn_factors.empty:

                st.info(
                    "No strong factors pushing toward churn."
                )

            else:

                for _, row in churn_factors.iterrows():

                    st.markdown(
                        f"""
                        <div class="factor-card">

                            <div class="factor-name">
                                {row["Feature"]}
                            </div>

                            <div class="factor-value">
                                Contribution toward churn:
                                {row["Contribution"]:.3f}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # ----------------------------------------------------
        # STAY FACTORS
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "### 🟢 Factors pushing toward staying"
            )

            if stay_factors.empty:

                st.info(
                    "No strong factors pushing toward staying."
                )

            else:

                for _, row in stay_factors.iterrows():

                    st.markdown(
                        f"""
                        <div class="factor-card">

                            <div class="factor-name">
                                {row["Feature"]}
                            </div>

                            <div class="factor-value">
                                Contribution toward staying:
                                {abs(row["Contribution"]):.3f}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # ====================================================
        # BUSINESS RECOMMENDATION
        # ====================================================

        st.markdown(
            '<div class="section-title">💼 Recommended Business Action</div>',
            unsafe_allow_html=True
        )


        if probability >= 0.5:

            st.warning("""
            **Retention action recommended.**

            This customer has high predicted churn risk.
            The company could contact the customer, investigate
            dissatisfaction, offer a suitable retention incentive,
            or encourage a longer-term contract.
            """)

        elif probability >= 0.3:

            st.info("""
            **Monitor this customer.**

            The customer has moderate churn risk.
            Monitor satisfaction and engagement to prevent
            future churn.
            """)

        else:

            st.success("""
            **Low immediate risk.**

            This customer currently has a relatively low
            predicted probability of churn.
            """)


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Performance of the Logistic Regression model "
        "on the available Telco Customer Churn dataset."
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.1f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.1f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.1f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.1f}%"
        )


    with col5:

        st.metric(
            "ROC-AUC",
            f"{auc:.3f}"
        )


    # ========================================================
    # CONFUSION MATRIX + ROC
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    with col1:

        st.markdown("### 🎯 Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred
        )

        fig, ax = plt.subplots()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Stay", "Churn"]
        )

        disp.plot(
            ax=ax,
            values_format="d"
        )

        ax.set_title("Confusion Matrix")

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    with col2:

        st.markdown("### 📈 ROC Curve")

        fpr, tpr, _ = roc_curve(
            y,
            y_prob
        )

        fig, ax = plt.subplots()

        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {auc:.3f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "ROC Curve"
        )

        ax.legend()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


    # ========================================================
    # FEATURE INFLUENCE
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 Feature Influence</div>',
        unsafe_allow_html=True
    )


    coefficients = model.coef_[0]


    feature_importance = pd.DataFrame({

        "Feature": model_columns,

        "Coefficient": coefficients

    })


    feature_importance = (
        feature_importance
        .sort_values(
            "Coefficient",
            ascending=False
        )
    )


    # Top 15
    top_features = feature_importance.head(15)


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    ax.barh(
        top_features["Feature"][::-1],
        top_features["Coefficient"][::-1]
    )


    ax.set_xlabel(
        "Logistic Regression Coefficient"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Top Features Influencing Churn"
    )


    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🤖 Model Information</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Algorithm",
            "Logistic Regression"
        )


    with col2:

        st.metric(
            "Features",
            len(model_columns)
        )


    with col3:

        st.metric(
            "Dataset Rows",
            len(df)
        )


    st.info("""
    **How to interpret the model:**

    Positive Logistic Regression coefficients push the
    prediction toward churn.

    Negative coefficients push the prediction toward
    staying.

    ROC-AUC measures how well the model separates
    customers who churn from customers who stay.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ChurnGuard • Logistic Regression • Customer Churn Prediction"
)
