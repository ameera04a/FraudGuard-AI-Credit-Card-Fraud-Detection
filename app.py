import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Custom CSS for better frontend
# -------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #555;
}
.card {
    background-color: #f7f9fc;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #1f4e79;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
.safe-box {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2e7d32;
}
.fraud-box {
    background-color: #ffebee;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #c62828;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load model and scaler
# -------------------------------
@st.cache_resource
def load_files():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_files()
except:
    model = None
    scaler = None

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("💳 FraudGuard AI")
st.sidebar.write("Credit Card Fraud Detection System")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict Transaction", "Model Results", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("Model: Logistic Regression\n\nBalancing: SMOTE\n\nFrontend: Streamlit")

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.markdown('<div class="main-title">FraudGuard AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A Machine Learning Based Credit Card Fraud Detection System</div>',
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Logistic Regression")

    with col2:
        st.metric("Dataset", "Credit Card Fraud")

    with col3:
        st.metric("Frontend", "Streamlit")

    st.write("")

    st.markdown("""
    <div class="card">
    <h3>Project Overview</h3>
    <p>
    FraudGuard AI is a machine learning project that predicts whether a credit card transaction
    is legitimate or fraudulent. The system uses transaction features, applies preprocessing,
    balances the dataset using SMOTE, and trains a Logistic Regression model.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.subheader("Project Workflow")

    st.code("""
Dataset Collection
        ↓
Data Preprocessing
        ↓
Train-Test Split
        ↓
SMOTE Balancing
        ↓
Logistic Regression Model Training
        ↓
Model Evaluation
        ↓
Streamlit Frontend Integration
    """)

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Predict Transaction":
    st.title("Predict Credit Card Transaction")

    if model is None or scaler is None:
        st.error("Model or scaler file not found. Please make sure fraud_model.pkl and scaler.pkl are in the same folder as app.py.")
    else:
        st.markdown("""
        Select a transaction from the dataset and test whether the model predicts it as
        **Legitimate** or **Fraudulent**.
        """)

        try:
            data = pd.read_csv("sample_transactions.csv")

            st.success("Dataset loaded successfully!")

            # Store selected row number in session state
            if "row_number" not in st.session_state:
                st.session_state.row_number = 0

            st.subheader("Choose a Transaction")

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("✅ Select Random Legitimate Transaction"):
                    st.session_state.row_number = data[data["Class"] == 0].sample(1).index[0]

            with col_b:
                if st.button("⚠️ Select Random Fraud Transaction"):
                    st.session_state.row_number = data[data["Class"] == 1].sample(1).index[0]

            row_number = st.number_input(
                "Transaction Row Number",
                min_value=0,
                max_value=len(data) - 1,
                value=int(st.session_state.row_number),
                step=1
            )

            st.session_state.row_number = row_number

            selected_row = data.iloc[int(row_number)]

            st.write("")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Transaction Time", round(selected_row["Time"], 2))

            with col2:
                st.metric("Transaction Amount", round(selected_row["Amount"], 2))

            with col3:
                actual_class = "Fraud" if selected_row["Class"] == 1 else "Legitimate"
                st.metric("Actual Class", actual_class)

            st.write("")

            with st.expander("View Full Transaction Details"):
                st.dataframe(selected_row.to_frame().T)

            if st.button("Predict Selected Transaction"):
                input_data = selected_row.drop("Class").to_frame().T

                # Scale Time and Amount using saved scaler
                input_data[["Time", "Amount"]] = scaler.transform(
                    input_data[["Time", "Amount"]]
                )

                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

                st.write("")

                predicted_class = "Fraud" if prediction == 1 else "Legitimate"

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric("Predicted Class", predicted_class)

                with result_col2:
                    st.metric("Fraud Probability", f"{probability * 100:.2f}%")

                if prediction == 1:
                    st.markdown(f"""
                    <div class="fraud-box">
                    <h2>⚠️ Fraudulent Transaction Detected</h2>
                    <p><b>Fraud Probability:</b> {probability * 100:.2f}%</p>
                    <p>The model predicts this transaction as suspicious.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-box">
                    <h2>✅ Legitimate Transaction</h2>
                    <p><b>Fraud Probability:</b> {probability * 100:.2f}%</p>
                    <p>The model predicts this transaction as safe.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")

                if actual_class == predicted_class:
                    st.success("✅ The model prediction matches the actual class.")
                else:
                    st.error("❌ The model prediction does not match the actual class.")

        except FileNotFoundError:
            st.error("creditcard.csv not found. Please place it in the same folder as app.py.")
# -------------------------------
# Model Results Page
# -------------------------------
elif page == "Model Results":
    st.title("Model Results and Visualizations")

    st.markdown("""
    This section presents the final model performance and shows how class imbalance was handled using SMOTE.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "99%+")
    col2.metric("Precision", "81%")
    col3.metric("Recall", "78%")
    col4.metric("F1-score", "79%")

    st.write("")

    # -------------------------------
    # Confusion Matrix and Metrics
    # -------------------------------
    st.subheader("Model Performance")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("##### Confusion Matrix")

        cm = np.array([[56634, 17],
                       [21, 74]])

        fig, ax = plt.subplots(figsize=(2.8, 2.2))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Legitimate", "Fraud"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        ax.set_title("Confusion Matrix", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual", fontsize=7)
        ax.tick_params(axis='both', labelsize=6)

        for text in display.text_.ravel():
            text.set_fontsize(7)

        st.pyplot(fig, use_container_width=False)

    with right_col:
        st.markdown("##### Model Performance Metrics")

        metrics = ["Precision", "Recall", "F1-score"]
        values = [0.81, 0.78, 0.79]

        fig2, ax2 = plt.subplots(figsize=(2.8, 2.2))

        ax2.bar(metrics, values)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Score", fontsize=7)
        ax2.set_title("Fraud Class Performance", fontsize=8)
        ax2.tick_params(axis='both', labelsize=6)

        for index, value in enumerate(values):
            ax2.text(index, value + 0.02, str(value), ha='center', fontsize=7)

        st.pyplot(fig2, use_container_width=False)

    st.write("")

    # -------------------------------
    # Before and After SMOTE Graphs
    # -------------------------------
    st.subheader("Class Imbalance Handling")

    st.info("""
    The original dataset was highly imbalanced because legitimate transactions were much higher than fraud transactions.
    SMOTE was applied on the training data to increase fraud samples from a very small amount to a more useful ratio.
    """)

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("##### Before SMOTE")

        labels = ["Legitimate", "Fraud"]
        before_counts = [283253, 473]
        before_percentages = [
            before_counts[0] / sum(before_counts) * 100,
            before_counts[1] / sum(before_counts) * 100
        ]

        fig3, ax3 = plt.subplots(figsize=(2.6, 2.1))

        ax3.bar(labels, before_percentages)
        ax3.set_title("Original Dataset Distribution", fontsize=8)
        ax3.set_xlabel("Transaction Type", fontsize=7)
        ax3.set_ylabel("Percentage (%)", fontsize=7)
        ax3.set_ylim(0, 105)
        ax3.tick_params(axis='both', labelsize=6)

        for index, value in enumerate(before_percentages):
            ax3.text(index, value + 1, f"{value:.2f}%", ha='center', fontsize=6)

        st.pyplot(fig3, use_container_width=False)

    with col_after:
        st.markdown("##### After SMOTE")

        labels = ["Legitimate", "Fraud"]
        after_counts = [226602, 4532]
        after_percentages = [
            after_counts[0] / sum(after_counts) * 100,
            after_counts[1] / sum(after_counts) * 100
        ]

        fig4, ax4 = plt.subplots(figsize=(2.6, 2.1))

        ax4.bar(labels, after_percentages)
        ax4.set_title("Training Data After SMOTE", fontsize=8)
        ax4.set_xlabel("Transaction Type", fontsize=7)
        ax4.set_ylabel("Percentage (%)", fontsize=7)
        ax4.set_ylim(0, 105)
        ax4.tick_params(axis='both', labelsize=6)

        for index, value in enumerate(after_percentages):
            ax4.text(index, value + 1, f"{value:.2f}%", ha='center', fontsize=6)

        st.pyplot(fig4, use_container_width=False)

    st.markdown("""
    **Before SMOTE:** Fraud transactions were almost invisible because they were only around **0.17%** of the dataset.  
    **After SMOTE:** Fraud transactions increased to around **1.96%** of the training data using `sampling_strategy=0.02`.
    """)

    st.warning("""
    Accuracy alone can be misleading in fraud detection because the dataset is highly imbalanced.
    Therefore, precision, recall, F1-score, and confusion matrix are also used for evaluation.
    """)
# -------------------------------
# About Project Page
# -------------------------------
elif page == "About Project":
    st.title("About the Project")

    st.markdown("""
    ### Problem Statement

    Credit card fraud is a serious issue in the financial sector. Since thousands of transactions
    happen daily, manually detecting fraudulent transactions is difficult. This project uses machine
    learning to predict whether a transaction is legitimate or fraudulent.

    ### Algorithm Used

    **Logistic Regression** was used because it is suitable for binary classification problems,
    where the output has two classes: fraud or legitimate.

    ### Dataset

    The project uses the Kaggle Credit Card Fraud Detection dataset. The dataset contains numerical
    transaction features such as Time, Amount, V1 to V28, and Class.

    ### Class Labels

    - `0` = Legitimate Transaction
    - `1` = Fraudulent Transaction

    ### Imbalanced Dataset Handling

    SMOTE was used to handle class imbalance. It creates synthetic fraud samples so the model can
    learn fraud patterns better.

    ### Learning Outcomes

    - Data preprocessing
    - Feature scaling
    - Train-test split
    - SMOTE balancing
    - Logistic Regression training
    - Model evaluation
    - Streamlit frontend integration
    """)