import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Custom CSS - light mode + dark mode friendly
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

.small-note {
    font-size: 14px;
    color: inherit;
    opacity: 0.85;
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
except Exception:
    model = None
    scaler = None

# -------------------------------
# Helper functions
# -------------------------------
def make_fraud_gauge(probability):
    fraud_percent = probability * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_percent,
        title={"text": "Fraud Probability (%)"},
        number={"suffix": "%", "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#c62828" if probability >= 0.5 else "#2e7d32"},
            "steps": [
                {"range": [0, 30], "color": "rgba(46, 125, 50, 0.25)"},
                {"range": [30, 70], "color": "rgba(255, 193, 7, 0.25)"},
                {"range": [70, 100], "color": "rgba(198, 40, 40, 0.25)"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))

    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_probability_bar(probability):
    prob_df = pd.DataFrame({
        "Class": ["Legitimate", "Fraud"],
        "Probability": [(1 - probability) * 100, probability * 100]
    })

    fig = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        text="Probability",
        title="Prediction Confidence by Class"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 100], yaxis_title="Probability (%)")
    return fig


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
st.sidebar.info("Model: Logistic Regression\n\nBalancing: SMOTE\n\nFrontend: Streamlit + Plotly")

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

    workflow_df = pd.DataFrame({
        "Step Number": [1, 2, 3, 4, 5, 6, 7],
        "Step": [
            "Dataset Collection",
            "Data Preprocessing",
            "Train-Test Split",
            "SMOTE Balancing",
            "Logistic Regression Training",
            "Model Evaluation",
            "Streamlit Frontend Integration"
        ]
    })

    fig_workflow = px.line(
        workflow_df,
        x="Step Number",
        y=[1] * len(workflow_df),
        text="Step",
        title="FraudGuard AI Workflow"
    )
    fig_workflow.update_traces(mode="markers+text", marker=dict(size=18), textposition="top center")
    fig_workflow.update_yaxes(visible=False)
    fig_workflow.update_layout(height=330, showlegend=False, xaxis=dict(dtick=1))
    st.plotly_chart(fig_workflow, use_container_width=True)

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
                st.dataframe(selected_row.to_frame().T, use_container_width=True)

            # Interactive amount comparison graph
            st.subheader("Selected Transaction Amount Comparison")
            amount_df = pd.DataFrame({
                "Type": ["Selected Transaction", "Average Sample Amount"],
                "Amount": [selected_row["Amount"], data["Amount"].mean()]
            })

            fig_amount = px.bar(
                amount_df,
                x="Type",
                y="Amount",
                text="Amount",
                title="Selected Amount vs Average Amount"
            )
            fig_amount.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig_amount, use_container_width=True)

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

                graph_col1, graph_col2 = st.columns(2)

                with graph_col1:
                    st.plotly_chart(make_fraud_gauge(probability), use_container_width=True)

                with graph_col2:
                    st.plotly_chart(make_probability_bar(probability), use_container_width=True)

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
            st.error("sample_transactions.csv not found. Please place it in the same folder as app.py.")

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
    # Interactive Model Performance
    # -------------------------------
    st.subheader("Model Performance")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("##### Interactive Confusion Matrix")

        cm = np.array([[56634, 17],
                       [21, 74]])

        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            title="Confusion Matrix"
        )
        cm_fig.update_layout(height=420)
        st.plotly_chart(cm_fig, use_container_width=True)

    with right_col:
        st.markdown("##### Interactive Fraud Class Metrics")

        metrics_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-score"],
            "Score": [0.81, 0.78, 0.79]
        })

        metrics_fig = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            text="Score",
            title="Fraud Class Performance"
        )
        metrics_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        metrics_fig.update_layout(yaxis_range=[0, 1], height=420, yaxis_title="Score")
        st.plotly_chart(metrics_fig, use_container_width=True)

    st.write("")

    # -------------------------------
    # Confusion Matrix Breakdown
    # -------------------------------
    st.subheader("Confusion Matrix Breakdown")

    breakdown_df = pd.DataFrame({
        "Outcome": ["True Legitimate", "False Alarm", "Missed Fraud", "Correct Fraud"],
        "Count": [56634, 17, 21, 74]
    })

    breakdown_fig = px.pie(
        breakdown_df,
        names="Outcome",
        values="Count",
        title="Prediction Outcome Breakdown",
        hole=0.35
    )
    st.plotly_chart(breakdown_fig, use_container_width=True)

    st.info("""
    The model correctly detected 74 fraud cases and missed 21 fraud cases in the test set.
    Since fraud detection is highly imbalanced, precision, recall, F1-score, and confusion matrix
    are more useful than accuracy alone.
    """)

    # -------------------------------
    # Before and After SMOTE Graphs
    # -------------------------------
    st.subheader("Class Imbalance Handling")

    st.info("""
    The original dataset was highly imbalanced because legitimate transactions were much higher than fraud transactions.
    SMOTE was applied on the training data to increase fraud samples from a very small amount to a more useful ratio.
    """)

    smote_df = pd.DataFrame({
        "Stage": ["Before SMOTE", "Before SMOTE", "After SMOTE", "After SMOTE"],
        "Class": ["Legitimate", "Fraud", "Legitimate", "Fraud"],
        "Count": [283253, 473, 226602, 4532]
    })

    smote_df["Percentage"] = smote_df.groupby("Stage")["Count"].transform(lambda x: x / x.sum() * 100)

    smote_fig = px.bar(
        smote_df,
        x="Stage",
        y="Percentage",
        color="Class",
        barmode="group",
        text="Percentage",
        title="Class Distribution Before and After SMOTE"
    )
    smote_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    smote_fig.update_layout(yaxis_title="Percentage (%)", yaxis_range=[0, 105])
    st.plotly_chart(smote_fig, use_container_width=True)

    # -------------------------------
    # Counts before and after SMOTE
    # -------------------------------
    counts_fig = px.bar(
        smote_df,
        x="Stage",
        y="Count",
        color="Class",
        barmode="group",
        text="Count",
        title="Transaction Counts Before and After SMOTE"
    )
    counts_fig.update_traces(textposition="outside")
    st.plotly_chart(counts_fig, use_container_width=True)

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

    ### Interactive Frontend Features

    - Random legitimate and fraud transaction selection
    - Fraud probability gauge
    - Prediction confidence chart
    - Interactive confusion matrix
    - SMOTE comparison graphs
    - Dark-mode friendly project overview cards

    ### Learning Outcomes

    - Data preprocessing
    - Feature scaling
    - Train-test split
    - SMOTE balancing
    - Logistic Regression training
    - Model evaluation
    - Streamlit frontend integration
    - Interactive Plotly visualizations
    """)
