import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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
    color: #1f77b4;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: inherit;
}
.card {
    background-color: rgba(120, 120, 120, 0.12);
    color: inherit;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #1f77b4;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.15);
}
.card h3, .card p {
    color: inherit;
}
.workflow-box {
    background-color: rgba(120, 120, 120, 0.10);
    color: inherit;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(120, 120, 120, 0.25);
    min-height: 92px;
}
.workflow-arrow {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #1f77b4;
    padding-top: 28px;
}
.safe-box {
    background-color: rgba(46, 125, 50, 0.18);
    color: inherit;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2e7d32;
}
.fraud-box {
    background-color: rgba(198, 40, 40, 0.18);
    color: inherit;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #c62828;
}
.safe-box h2, .safe-box p,
.fraud-box h2, .fraud-box p {
    color: inherit;
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

    w1, a1, w2, a2, w3 = st.columns([2, 0.4, 2, 0.4, 2])
    with w1:
        st.markdown('<div class="workflow-box"><b>1. Dataset</b><br>Credit card transactions with Time, Amount, V1-V28, and Class.</div>', unsafe_allow_html=True)
    with a1:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
    with w2:
        st.markdown('<div class="workflow-box"><b>2. Preprocessing</b><br>Scale Time and Amount, then split data into training and testing.</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
    with w3:
        st.markdown('<div class="workflow-box"><b>3. SMOTE</b><br>Increase fraud samples in training data to handle class imbalance.</div>', unsafe_allow_html=True)

    st.write("")

    w4, a3, w5, a4, w6 = st.columns([2, 0.4, 2, 0.4, 2])
    with w4:
        st.markdown('<div class="workflow-box"><b>4. Model Training</b><br>Train Logistic Regression on balanced training data.</div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
    with w5:
        st.markdown('<div class="workflow-box"><b>5. Evaluation</b><br>Use confusion matrix, precision, recall, and F1-score.</div>', unsafe_allow_html=True)
    with a4:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
    with w6:
        st.markdown('<div class="workflow-box"><b>6. Frontend</b><br>Streamlit app predicts selected transactions and shows results.</div>', unsafe_allow_html=True)

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
    The graphs here are interactive, so you can hover over values and compare results easily.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "99%+")
    col2.metric("Precision", "81%")
    col3.metric("Recall", "78%")
    col4.metric("F1-score", "79%")

    st.write("")

    # -------------------------------
    # Data used for charts
    # -------------------------------
    cm = np.array([[56634, 17],
                   [21, 74]])

    tn, fp = cm[0]
    fn, tp = cm[1]

    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score"],
        "Score": [0.81, 0.78, 0.79]
    })

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.subheader("Model Performance")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("##### Interactive Confusion Matrix")

        fig_cm = px.imshow(
            confusion_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title="Confusion Matrix"
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            height=430
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with right_col:
        st.markdown("##### Fraud Class Performance")

        fig_metrics = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            text="Score",
            title="Precision, Recall, and F1-score"
        )
        fig_metrics.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_metrics.update_layout(
            yaxis_range=[0, 1],
            yaxis_title="Score",
            height=430
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    st.write("")

    # -------------------------------
    # Confusion matrix explanation graph
    # -------------------------------
    st.subheader("Prediction Breakdown")

    breakdown_df = pd.DataFrame({
        "Outcome": [
            "Correct Legitimate",
            "False Fraud Alarm",
            "Missed Fraud",
            "Correct Fraud"
        ],
        "Count": [tn, fp, fn, tp],
        "Meaning": [
            "Legitimate transactions correctly detected",
            "Legitimate transactions predicted as fraud",
            "Fraud transactions predicted as legitimate",
            "Fraud transactions correctly detected"
        ]
    })

    fig_breakdown = px.bar(
        breakdown_df,
        x="Outcome",
        y="Count",
        text="Count",
        hover_data=["Meaning"],
        title="Confusion Matrix Breakdown"
    )
    fig_breakdown.update_traces(textposition="outside")
    fig_breakdown.update_layout(
        yaxis_title="Number of Transactions",
        xaxis_title="Prediction Outcome",
        height=430
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

    st.info("""
    This chart separates the confusion matrix into four practical outcomes. It helps explain how many frauds were correctly detected,
    how many frauds were missed, and how many legitimate transactions were incorrectly flagged.
    """)

    st.write("")

    # -------------------------------
    # Before and after SMOTE interactive graph
    # -------------------------------
    st.subheader("Class Imbalance Handling with SMOTE")

    st.info("""
    The original dataset was highly imbalanced because legitimate transactions were much higher than fraud transactions.
    SMOTE was applied only on the training data to increase fraud samples from a very small amount to a more useful ratio.
    """)

    smote_df = pd.DataFrame({
        "Stage": ["Before SMOTE", "Before SMOTE", "After SMOTE", "After SMOTE"],
        "Class": ["Legitimate", "Fraud", "Legitimate", "Fraud"],
        "Count": [283253, 473, 226602, 4532]
    })

    smote_df["Percentage"] = smote_df.groupby("Stage")["Count"].transform(lambda x: x / x.sum() * 100)

    fig_smote = px.bar(
        smote_df,
        x="Stage",
        y="Percentage",
        color="Class",
        barmode="group",
        text="Percentage",
        hover_data={"Count": True, "Percentage": ":.4f"},
        title="Class Distribution Before and After SMOTE"
    )
    fig_smote.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_smote.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title="Training Stage",
        yaxis_range=[0, 105],
        height=450
    )
    st.plotly_chart(fig_smote, use_container_width=True)

    # -------------------------------
    # Log scale chart to handle very large legitimate class
    # -------------------------------
    st.markdown("##### Class Counts on Log Scale")

    fig_log = px.bar(
        smote_df,
        x="Stage",
        y="Count",
        color="Class",
        barmode="group",
        text="Count",
        log_y=True,
        title="Class Counts Before and After SMOTE (Log Scale)"
    )
    fig_log.update_traces(textposition="outside")
    fig_log.update_layout(
        yaxis_title="Number of Transactions (Log Scale)",
        xaxis_title="Training Stage",
        height=450
    )
    st.plotly_chart(fig_log, use_container_width=True)

    st.markdown("""
    **Before SMOTE:** Fraud transactions were almost invisible because they were only around **0.17%** of the dataset.  
    **After SMOTE:** Fraud transactions increased to around **1.96%** of the training data using `sampling_strategy=0.02`.  
    The log-scale graph is used because the legitimate class is extremely large compared to the fraud class.
    """)

    st.write("")

    # -------------------------------
    # Model workflow graph on results page
    # -------------------------------
    st.subheader("Model Workflow Summary")

    workflow_steps = pd.DataFrame({
        "Step": [
            "Dataset",
            "Preprocessing",
            "Train-Test Split",
            "SMOTE",
            "Training",
            "Evaluation",
            "Frontend"
        ],
        "Order": [1, 2, 3, 4, 5, 6, 7],
        "Description": [
            "Credit card transaction data with Time, Amount, V1-V28, and Class.",
            "Scale Time and Amount so the model receives normalized values.",
            "Split data into training and testing sets.",
            "Apply SMOTE on training data to increase fraud examples.",
            "Train Logistic Regression model on balanced data.",
            "Evaluate using confusion matrix, precision, recall, and F1-score.",
            "Use Streamlit to test transactions through the frontend."
        ]
    })

    fig_workflow = go.Figure()

    fig_workflow.add_trace(go.Scatter(
        x=workflow_steps["Order"],
        y=[1] * len(workflow_steps),
        mode="markers+text+lines",
        marker=dict(size=38),
        text=workflow_steps["Step"],
        textposition="top center",
        hovertext=workflow_steps["Description"],
        hoverinfo="text"
    ))

    fig_workflow.update_layout(
        title="FraudGuard AI Pipeline",
        xaxis=dict(
            tickmode="array",
            tickvals=workflow_steps["Order"],
            ticktext=workflow_steps["Step"],
            showgrid=False
        ),
        yaxis=dict(visible=False),
        height=330,
        showlegend=False
    )

    st.plotly_chart(fig_workflow, use_container_width=True)

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
